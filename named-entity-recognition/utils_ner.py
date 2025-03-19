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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import json
import logging
import os
import pdb
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from tqdm import tqdm

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_torch_available
from datasets import load_dataset

import pandas as pd

logger = logging.getLogger(__name__)

import os
import csv
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from filelock import FileLock
import logging



logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """


    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "validation"
    test = "test"
    # train = ['train','validation']


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset


    class NerDataset(Dataset):
        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

        def __init__(
            self,
            data_source: str,
            output_dir: Optional[str],
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = 128,  # Default value
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                os.path.dirname(data_source), "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_source}")
                    
                    # if isinstance(mode, Split):
                    #     mode = mode.value
                    if data_source.endswith('.csv'):
                        examples=read_examples_from_csv(data_path=data_source,split=mode)
                    elif data_source.endswith('.txt'):
                        examples=read_examples_from_file(data_dir=data_source,mode=mode)
                    elif data_source.endswith('.json'):
                        examples=read_examples_from_json(data_path=data_source,split=mode)
                    elif data_source.endswith('.jsonl'):
                        examples=read_examples_from_jsonl(data_path=data_source,split=mode)
                    else: 
                        if output_dir is not None:
                            examples = read_examples_from_huggingface(dataset_name=data_source,output=output_dir, split=mode)
                        else:
                            examples = read_examples_from_huggingface(dataset_name=data_source, split=mode)

                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)



        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]
        

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    splits_replace = splits[-1].replace("\n", "")
                    if splits_replace == 'O':
                        labels.append(splits_replace)
                    else:
                        labels.append(splits_replace + "-bio")
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples

# Yashfinul Haque 7/18/24
# Dataloading Util Functions
def read_examples_from_csv(data_path:str, split:Split) -> List[InputExample]:
    df = pd.read_csv(data_path)
    mode = split.value if isinstance(split,Split) else split
    examples = []

    if mode == 'test':
        examples = [
            InputExample(
                guid=row["index"],
                words=row["sentence"].split(),
                labels= [0] * len(row["sentence"].split())
            )
            for _,row in df.iterrows()
        ]
    else:
        examples = [
            InputExample(
                guid=row["index"],
                words=row["sentence"].split(),
                labels=row["ner_tags"]
            )
            for _,row in df.iterrows()
        ]

    return examples

    
def read_examples_from_json(data_path:str, split:Split) -> List[InputExample]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mode = split.value if isinstance(split,Split) else split
    example = []

    if mode == 'test':
        examples = [
            InputExample(
                guid=item["index"],
                words=item["sentence"].split(),
                labels= [0] * len(item["sentence"].split())
            )
            for item in data
        ]
    else:
        examples = [
            InputExample(
                guid=item["index"],
                words=item["sentence"].split(),
                labels=item["ner_tags"]
            )
            for item in data
        ]
    
    return examples


def read_examples_from_jsonl(data_path:str, split:Split) -> List[InputExample]:
    examples=[]
    mode = split.value if isinstance(split, Split) else split

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            if mode == "test":
                words = item["sentence"].split()
                examples.append(InputExample(
                    guid=item["index"],
                    words=words,
                    labels=[0] * len(words)
                ))
            else:
                examples.append(InputExample(
                    guid=item["index"],
                    words=item["sentence"].split(),
                    labels=item["ner_tags"]
                ))
    
    return examples

def read_examples_from_huggingface(dataset_name:str,output:Optional[str]=None, split:Split=Split.train) -> List[InputExample]:
    mode = split.value if isinstance(split, Split) else split
    examples = []
    
    if isinstance(mode, list):
        if output is not None: 
            if not os.path.exists(output):
                os.makedirs(output)
            for m in mode:
                dataset = load_dataset(dataset_name, data_dir=output, split=m)
                for item in dataset:
                    examples.append(InputExample(
                        guid=item["id"],
                        words=item["tokens"],
                        labels=item["ner_tags"]
                    ))
    else:
        dataset = load_dataset(dataset_name, split=mode)
        for item in dataset:
            examples.append(InputExample(
                guid=item["id"],
                words=item["tokens"],
                labels=item["ner_tags"]
            ))

    return examples
# **************************END****************************************

# Featurization
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    label_scheme:bool = False,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    
    # Flatten the label_list, if any exists
    if any(isinstance(i, list) for i in label_list):
        label_list = [label for sublist in label_list for label in sublist]

    # Create label_map from flattened label_list
    if label_scheme:
        label_map = {label: i for i, label in enumerate(label_list)}
    # Initialize Features List
    features = []
    
    # Process each example:
    for ex_index, example in enumerate(tqdm(examples, desc="Processing examples")):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # Tokenize Words and Map Labels:
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                if label_scheme:
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                else:
                    label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Truncate sequences if necessary
        # This block truncates tokens and labels to ensure the sequence length
        # does not exceed `max_seq_length`
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        
        # Add class[CLS] token based on cls_token_at_end
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create Attention Mask
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Pad Sequences
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        # Assertions to ensure correct lengths
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # Logging
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        
        # Handle token type IDs
        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        # Append the features
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    return features


def get_labels() -> List[str]:
    return ['O','B-DNA','I-DNA','B-RNA','I-RNA','B-cell_line','I-cell_line','B-cell_type','I-cell_type','B-protein','I-protein']
