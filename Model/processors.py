import logging
import os
from  sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
import csv
import re
import pickle
import pandas as pd
import numpy as np
import dataclasses
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union
logger = logging.getLogger(__name__)

class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()


    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_examples(self, data_dir=None, args=None):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

@dataclass
class InputExample:
    """
    A single training/test example
    """
    guid: str
    findings: str=None
    images: List[str]=None
    multi_task_labels: Optional[List[str]] = None
    multi_labels : Optional[List[str]] = None
    identifier: Optional[str]=None
    dateTime : Optional[str]=None
    originalFileName: Optional[str]=None
    finger : int=None
    label :int = None
    multi_labels_with_binary:Optional[List[str]]=None
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    img: Iterable[float]= None
    identifier:Optional[str]=None
    multi_task_labels:Optional[Union[int, float]] = None
    multi_labels:Optional[Union[int, float]] = None

    finger_index:Optional[int]= None
    label :Optional[int]= None
    multi_labels_with_binary: Optional[List[str]]=None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class NailImageProcessor(DataProcessor):

    """
    """
    def __init__(self, args):

        self.data_dir= args.data_dir
        self.args = args

        self.class_names = ["finger_dilatierte", "finger_riesen", "finger_rare", "finger_mikro", "finger_Dysang"]
        #self.class_names = [ "finger_riesen", "finger_rare", "finger_Dysang"]
        ''' 
        finger_dilatierte: 'enlarged capillaries', 
        finger_riesen':'giant capillaries'
        finger_rare' :'capillary loss',
        'finger_mikro': 'microhaemorrhages'
        '''

        self.label2id = {
            0: '0',
            1: '+',
            2: '++',
            3: '+++',
        }

        self.id2lable = {v:k for k,v in self.label2id.items()}
        self.finge2id={
            "2li": 0,
            "3li": 1,
            "4li": 2,
            "5li": 3,
            "2re": 4,
            "3re": 5,
            "4re": 6,
            "5re": 7,
        }

        self.class2id = {class_name : idx for idx, class_name in enumerate(self.class_names)}

        self.multi_labels_ids = self.class2id.keys()

        """Creates examples for the training, dev and test sets."""

        """
        Image Index,Finding Label
        """

        self.data_cache_dir = args.data_cache_dir if args.data_cache_dir is not None else args.data_dir
        self.cached_examples_file = os.path.join(self.data_cache_dir, 'examples.pickle')

        #self.cached_examples_file_train = os.path.join(self.data_cache_dir, 'examples_train.pickle')
        #self.cached_examples_file_test = os.path.join(self.data_cache_dir, 'examples_test.pickle')
        #self.cached_examples_file_dev = os.path.join(self.data_cache_dir, 'examples_dev.pickle')


    def get_train_examples(self):
        examples = []
        cuda_path=os.path.join(self.args.data_dir, "intermediate_files", "2021-01-26_dataframe_input_model.csv")
        if os.path.exists(cuda_path):
            df = pd.read_csv(cuda_path, index_col=0)
        else:
            df = pd.read_csv(os.path.join(self.args.data_dir, "dataframe_input_model.csv"), index_col=0)

        for index, row in df.iterrows():
            idx = index,
            identifier = row['IDENTIFIER']
            examinationDateTime=row['ExaminationDateTime']
            fileName=row['FileName']
            originalFileName=row['OriginalFileName']
            finger = self.finge2id[row['finger']]
            try:
                multi_task_labels = [self.id2lable[l] for l in [row[fi] for fi in self.class_names]]
            except:
               continue

            #=IF(OR(NOT(ISBLANK(V5)),COUNTIF(W5,{"*++*";"*xx*"})=1),1,0)
            scleroderma_pattern = 0
            if row['finger_riesen']!='0' or row['finger_rare'] in ['++','+++']:
                scleroderma_pattern = 1


            multi_labels = [0 if l == 0 else 1 for l in multi_task_labels]
            guid = "%s-%s" % (idx, identifier)

            # image_path=os.path.join(self.data_dir,"images-224",image_path)
            cuda_path=os.path.join(self.data_dir,"content","images")
            if os.path.exists(cuda_path):
                if not os.path.exists(os.path.join(self.data_dir,"content","images", fileName)):
                    continue
            else:
                if not os.path.exists(os.path.join(self.data_dir,"images", fileName)):
                    continue


            examples.append (InputExample(guid=guid,
                                         identifier=identifier,
                                         dateTime=examinationDateTime,
                                         originalFileName=originalFileName,
                                         images=[fileName],
                                         multi_task_labels=multi_task_labels,
                                         multi_labels=multi_labels,
                                         finger=finger,
                                         label= 1 if 1 in multi_labels else 0,
                                         #multi_labels_with_binary=multi_labels +[1 if 1 in multi_labels else 0],
                                         multi_labels_with_binary=multi_labels + [scleroderma_pattern],
                                              )
                                 )

        # with open(self.cached_examples_file, 'wb') as f:
        #     pickle.dump(examples, f)
        return examples


    def get_test_examples(self):
       pass


    def get_dev_examples(self):
        pass

    def get_labels(self):
        return self.class2id


image_processors = {
    "NailImages":NailImageProcessor,
}