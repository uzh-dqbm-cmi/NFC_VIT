


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning XrayTransformer.
    Adapted from `examples/text-classification/run_glue.py`
"""

import sys
sys.path.append("..")

import argparse
import glob
import logging
import os, math
import random
import pandas as pd
import sklearn

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.model_selection import GroupShuffleSplit

from Model.classifier import ClassifierClass
from Model.processors import image_processors as processors
from Model.datasets import image_datasets as datasets

from collections import defaultdict
from Model.evaluation import multi_task_metrics,multi_label_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME,  AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, imageDataset, evaluate=False, mode=None):
    mode= mode if evaluate else "train"
    dataset= imageDataset(args=args, mode=mode, limit_length=args.limit_length)

    return dataset, np.array([f.identifier for f in dataset.features])

def train(args, train_dataset, model, label2id, dev_dataset=None, cv_idx=0):

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    #from torch.optim.lr_scheduler import ReduceLROnPlateau
    #scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')


    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path is not None
        and  os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt") )
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path is not None:
        # set global_step to global_step of last saved checkpoint from model path
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = float('-inf')
    best_loss = float('inf')
    patience_count = 0



    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        loss_cnt = 0
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # IF MULTITASK : batch[-1] else batch[1]
            if args.multiTask:
                label_ids = batch[-1]
            else:
                label_ids = batch[1]

            num_tasks=len(label_ids)

            #print(batch[0].device, weights.device, pos_weights.device)
            inputs = {"img": batch[0],
                      "labels": label_ids,
                      }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.multiTask:
                loss = sum(loss.values()) / num_tasks
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            epoch_loss+=loss.item()

            tr_loss += loss.item()

            loss_cnt += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results , _= evaluate(args, dev_dataset, model, label2id, meta_data=None, prefix="")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                   # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            # and epoch_num >10
        if args.early_stopping:
            results, dev_loss = evaluate(args, dev_dataset, model, label2id, prefix="dev")

            # multi-task
            if args.multiTask:
                dev_accuracy_auc = results['eval_accuracy']
            else:
                dev_accuracy_auc = results['eval_auc']

            if dev_loss < best_loss:
                patience_count = 0
                best_loss = dev_loss
                # save  model
                print(
                    '(loss: %.4f,fold: %d, epoch: %d, dev acc/auc = %.4f, dev loss =  %.4f), saving...' %
                    (epoch_loss / loss_cnt,
                     cv_idx,
                     epoch_num,
                     dev_accuracy_auc,
                     dev_loss
                     ))

                # Save model checkpoint
                # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                if not os.path.exists(os.path.join(args.output_dir,str(cv_idx))):
                    os.makedirs(os.path.join(args.output_dir,str(cv_idx)))
                logger.info("Saving model checkpoint to %s %s", args.output_dir, str(cv_idx))
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir,str(cv_idx), "pytorch_model.bin"))
                torch.save(args, os.path.join(args.output_dir,str(cv_idx), "training_args.bin"))
            else:
                print('(loss: %.4f, fold: %d, epoch: %d, dev acc.auc = %.4f, dev loss =  %.4f), without saving...' %
                      (epoch_loss / loss_cnt,
                       cv_idx,
                       epoch_num,
                       dev_accuracy_auc,
                       dev_loss
                       ))
                patience_count += 1

            if patience_count >= args.patience:
                train_iterator.close()
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_loss


def evaluate(args, eval_dataset, model, label2id,  prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = {} if args.multiTask else []
        labels = {} if args.multiTask else []

        all_logits = {t: [] for t in label2id}
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            if args.multiTask:
                label_ids = batch[-1]
            else:
                label_ids = batch[1]

            bs, n_crops, c, h, w = batch[0].size()
            with torch.no_grad():
                num_tasks = label_ids.size()[-1]

                inputs = {"img": batch[0].view(-1,c,h,w),
                          "labels": label_ids,
                          "n_crops":n_crops,
                          "batch_size":bs}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if args.multiTask:
                    tmp_eval_loss = sum(tmp_eval_loss.values()) / num_tasks
                    logits = {t: lt.detach().cpu().numpy() for t, lt in logits.items()}
                    label_ids = {t: batch[-1][:, i] for i, t in enumerate(label2id)}
                    label_ids = {t: l.to('cpu').numpy() for t, l in label_ids.items()}
                else:
                    logits=[lt.detach().cpu().numpy() for lt in logits]
                    label_ids=[l.to('cpu').numpy() for  l in label_ids]


                if len(preds) == 0:
                    if args.multiTask:
                        for task in label_ids.keys():
                            preds[task] = logits[task].tolist()
                            labels[task] = label_ids[task].tolist()
                    else:
                        preds=logits
                        labels=label_ids
                else:
                    if args.multiTask:
                        for task in label_ids.keys():
                            preds[task].extend(logits[task].tolist())
                            labels[task].extend(label_ids[task].tolist())
                    else:
                            preds.extend(logits)
                            labels.extend(label_ids)
                #tmp_eval_accuracy, rs = multi_task_metrics(logits, label_ids)
                if args.multiTask
                    eval_loss += tmp_eval_loss.item()
                else:
                    eval_loss += tmp_eval_loss.mean().item()

                #eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += batch[0].size(0)
                nb_eval_steps += 1
        if args.multiTask:
            eval_accuracy = multi_task_metrics(preds, labels)
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy}
        else:
            _, eval_auc=multi_label_metrics(preds, labels, label2id)
            result = {'eval_loss': eval_loss,
                       'eval_auc': eval_auc}

        eval_loss = eval_loss / nb_eval_steps
        results.update(result)


        output_eval_file = os.path.join(eval_output_dir, "eval_results_{}.txt".format(prefix))
        if os.path.exists(output_eval_file):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open(output_eval_file, append_write) as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            if args.multiTask:
                for task in labels.keys():
                    _preds, _labels = preds[task], labels[task]
                    _preds = np.argmax(_preds, axis=1)
                    y_actul = pd.Series(_labels, name='Actual')
                    y_pred = pd.Series(_preds, name='Predicted')
                    writer.write("***** Confusion Matrix for  {} *****".format(task))
                    df_confusion = pd.crosstab(y_actul, y_pred, rownames=['Actual'], colnames=['Predicted'],
                                               margins=True)
                    writer.write(df_confusion.to_string())
            else:



                preds=[sigmoid_v(p) for p in preds]
                preds=[(p >= 0.5).astype(int) for p in preds]
                cm = sklearn.metrics.multilabel_confusion_matrix(labels, preds)
                writer.write(np.array2string(cm))
                writer.write(sklearn.metrics.classification_report(labels,preds,target_names=label2id.keys()))

    return results, eval_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name",
                        default='NailImages',
                        type=str,
                        help={"help": "specify the task name: chesXray-14, cheXpert "},
                        )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--data_cache_dir",
        default=None,
        type=str,
        help="The processed data will be store in this path.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default="",
        type=str,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )


    parser.add_argument(
        "--img_dim",
        default=1024,
        type=int,
        help="image dimension of image encoder",
    )

    parser.add_argument(
        "--num_class",
        default=6,
        type=int,
        help="num classes on image encoder knowledge graph of image encoder",

    )



    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--early_stopping", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--patience", type=int, default=2, help="patience for early stopping")
    parser.add_argument("--limit_length", default=None ,type=int, help="number of examples to test the algo")
    parser.add_argument("--baseline", action="store_true", help="Freeze the visual backbone")
    parser.add_argument("--multiTask", action="store_true", help="Freeze the visual backbone")

    parser.add_argument("--autoAugment", action="store_true", help="Whether touse autoAugment or not.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    # Training
    if args.data_cache_dir is not None:
        if not os.path.exists(args.data_cache_dir):
            os.makedirs(args.data_cache_dir)

    processor = processors[args.task_name](args)
    imageDataset = datasets[args.task_name]

    label_list = processor.class2id
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    classifier=ClassifierClass["multi-task" if args.multiTask else "multi-label"]




    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab



    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        image_datasets, groups = load_and_cache_examples(args, imageDataset=imageDataset)
        gss = GroupShuffleSplit(n_splits=5, train_size=.6, random_state=42)
        results_cv = defaultdict(list)
        for cv_idx, (train_dev_idx, test_idx) in enumerate(gss.split(range(len(image_datasets)), groups=groups)):
            if args.model_name_or_path:
                model = classifier(num_labels_per_task={c: 4 for c in label_list.keys()})

                if not args.baseline:
                    logger.warning("You are instantiating from pretrained model.")
                    model_dict = model.state_dict()
                    pretrained_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'),
                                                 map_location=device)
                    # 1. filter out unnecessary keys
                    # common_keys=[k for k in model_dict.keys() if k in pretrained_dict.keys()] # if not from CheXpert pretrian model
                    common_keys = [k for k in model_dict.keys() if k in pretrained_dict.keys() if
                                   'visual_features.densenet121.classifier' not in k]

                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in common_keys}
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    model.load_state_dict(model_dict)
                    logger.info(f"model:{model.__class__.__name__} weights were initialized from {args.model_name_or_path}.\n")

            else:
                model = classifier(num_labels_per_task={c: 4 for c in label_list.keys()})

            model.to(args.device)
            logger.info(" Cross-Validation: %s", cv_idx)
            # for dev
            sub_gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
            train_dev_dataset = image_datasets.select_from_indices(list(train_dev_idx), mode='train')
            sub_groups= np.array([f.identifier for f in train_dev_dataset.features])

            train_idx, dev_idx = next(sub_gss.split(range(len(train_dev_dataset)),groups=sub_groups))
            train_dataset = train_dev_dataset.select_from_indices(list(train_idx), mode='train')
            dev_dataset = train_dev_dataset.select_from_indices(list(dev_idx), mode='dev')

            global_step, tr_loss , best_loss= train(args, train_dataset, model, label_list, dev_dataset=dev_dataset,cv_idx=cv_idx)
            logger.info(" global_step = %s, average loss = %s, best loss on dev= %s", global_step, tr_loss, best_loss)

            # Evaluation
            results = {}
            if args.do_eval and args.local_rank in [-1, 0]:
                checkpoints =  [os.path.join(args.output_dir,str(cv_idx))]
                if args.eval_all_checkpoints:
                    checkpoints = list(
                        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                    )
                    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
                logger.info("Evaluate the following checkpoints: %s", checkpoints)

                for checkpoint in checkpoints:
                    global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                    prefix = f'test-set_{cv_idx}'

                    model = classifier(num_labels_per_task={c:4 for c in label_list.keys()})
                    model_dict = model.state_dict()
                    pretrained_dict = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'))
                    # 1. filter out unnecessary keys
                    common_keys = [k for k in model_dict.keys() if k in pretrained_dict.keys()]
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in common_keys}
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    model.load_state_dict(model_dict)
                    logger.info(
                        f"model:{model.__class__.__name__} weights were initialized from {checkpoint}.\n")
                    model.to(args.device)
                    eval_dataset = image_datasets.select_from_indices(list(test_idx), mode='test')
                    result, _= evaluate(args, eval_dataset, model, label_list, prefix=prefix)
                    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                    results.update(result)
                    print(results)
            results_cv[cv_idx].append(results)
        print(results_cv)
    return results_cv

if __name__ == "__main__":
    main()




