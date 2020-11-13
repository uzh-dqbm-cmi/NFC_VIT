
import logging
import os
import time
from enum import Enum
from typing import List, Optional, Union
from tqdm import tqdm
from filelock import FileLock
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .autoaugment import ImageNetPolicy
from .processors import InputFeatures,InputExample,image_processors
import copy


logger = logging.getLogger(__name__)



IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD = [0.229, 0.224, 0.225]
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
NORMALIZAE = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)

# ten crop in test and validation provide 10 more view of image which it seems not correct to do in testing
TRANSFORM_TEST_TEN_CROP=transforms.Compose([
    transforms.Resize(512),
    transforms.TenCrop(512),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([NORMALIZAE(crop) for crop in crops])),
    ])

TRANSFORM_TRAIN=transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    ImageNetPolicy(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)
    ])

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    all = "all-in-one"

def image_to_tensor_ten_crop_auto_augment(img_path, mode=None):
    im_rgb = Image.open(img_path).convert('RGB')
    img_trs = TRANSFORM_TRAIN(im_rgb) if (mode == Split.train) else TRANSFORM_TEST_TEN_CROP(im_rgb)
    #print(img_trs.size())
    return img_trs

class NailImagesDataset(Dataset):
    """
    Create dataset from chestXray -14
    """

    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        #@todo update needed as data_dir
        args,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
    ):
        self.args = args
        processor = image_processors[args.task_name](args)

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        self.mode=mode

        cached_features_file = os.path.join(
            args.data_cache_dir if args.data_cache_dir is not None else args.data_dir,
            "cached_{}_{}".format(
                mode.value, args.task_name,
            ),
        )


        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) :
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_cache_dir if args.data_cache_dir is not None else args.data_dir}")
                if mode == Split.dev:
                    examples = processor.get_dev_examples()
                elif mode == Split.test:
                    examples = processor.get_test_examples()
                elif mode == Split.train:
                    examples = processor.get_train_examples()
                else:
                    examples= processor.get_examples()

                if limit_length is not None:
                    examples = examples[:limit_length]

                self.features = self.xray_convert_examples_to_features(examples)
                start = time.time()


                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature=self.features[i]
        #@todo if we have only one image?
        img = image_to_tensor_ten_crop_auto_augment(feature.img, mode=self.mode)
        multi_labels = torch.tensor(feature.multi_labels, dtype=torch.long)
        multi_task_labels = torch.tensor(feature.multi_task_labels, dtype=torch.long)
        return img, multi_labels, multi_task_labels


    def xray_convert_examples_to_features(self,
                                           examples: List[InputExample],
                                           ):

        features = []
        for example_index, example in tqdm(enumerate(examples)):
            if example_index % 10000 == 0:
                logger.info("Writing example %d" % (example_index))

            #@todo filter the report where it has single image !
            input_images = example.images
            # @todo how to detect which image is lateral and which one is frontal
            img = os.path.join(self.args.data_dir,"images", input_images[0])
            # #if self.mode==Split.train:
            # if self.mode==Split.train and not self.args.autoAugment:
            #     img1= image_to_tensor_ten_crop(img1, self.mode)
            inputs = {"identifier":example.identifier,
                      "img": img,
                      "multi_labels": example.multi_labels,
                      "multi_task_labels":example.multi_task_labels
                      }
            feature = InputFeatures(**inputs)
            features.append(feature)

        return features


    def select_from_indices(self, indices, mode=None):
        new_dataset= copy.deepcopy(self)
        if mode is not None:
           new_dataset.mode=Split[mode]
        new_dataset.features=[self.features[idx] for idx in indices]
        return  new_dataset

image_datasets = {
    "NailImages": NailImagesDataset,
}