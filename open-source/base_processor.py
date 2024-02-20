import os
import json
import torch
import random
import logging

import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self,
                 example_id,
                 input,
                 instruction=None,
                 instance=None,
                 labels=None,
                 **kwargs):
        """Constructs a EDInputExample.

        Args:
            example_id: Unique id for the example.
            text: List of str. The untokenized text.
            trigger_left: Left position of trigger.
            trigger_right: Light position of tigger.
            labels: Event type of the trigger
        """
        self.example_id = example_id
        self.input = input
        self.labels = labels
        self.kwargs = kwargs
        self.instruction = instruction
        self.instance = instance


class InputFeatures(object):
    def __init__(self,
                 example_id: Union[int, str],
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: Optional[List[int]] = None,
                 labels: Optional[List[int]] = None) -> None:
        """Constructs an `EDInputFeatures`."""
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels



class DataProcessor(Dataset):
    def __init__(self,
                 config,
                 tokenizer,
                 is_train_set) -> None:
        """Constructs an `EDDataProcessor`."""
        self.config = config
        self.tokenizer = tokenizer
        self.is_train_set = is_train_set
        self.examples = []
        self.input_features = []

    def read_examples(self,
                      input_file: str):
        """Obtains a collection of `EDInputExample`s for the dataset."""
        raise NotImplementedError

    def convert_examples_to_features(self):
        """Converts the `EDInputExample`s into `EDInputFeatures`s."""
        raise NotImplementedError

    def _truncate(self,
                  outputs: dict,
                  max_seq_length: int):
        """Truncates the sequence that exceeds the maximum length."""
        is_truncation = False
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation

    def get_ids(self) -> List[Union[int, str]]:
        """Returns the id of the examples."""
        ids = []
        for example in self.examples:
            ids.append(example.example_id)
        return ids

    def get_labels(self) -> List[List[str]]:
        all_labels = []
        for example in self.examples:
            all_labels.append(example.labels)
        return all_labels

    def __len__(self) -> int:
        """Returns the length of the examples."""
        return len(self.input_features)

    def __getitem__(self,
                    index: int) -> Dict[str, torch.Tensor]:
        """Obtains the features of a given example index and converts them into a dictionary."""
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collates the samples in batches."""
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            if self.config.truncate_seq2seq_output: # Seq2Seq
                for key in ["input_ids", "attention_mask", "token_type_ids"]:
                    if key not in output_batch:
                        continue
                    output_batch[key] = output_batch[key][:, :input_length]
                output_length = int((output_batch["labels"] != -100).sum(-1).max())
                output_batch["labels"] = output_batch["labels"][:, :output_length]
            else: # CausalLM left padding
                seq_start = self.config.max_seq_length - input_length
                for key in ["input_ids", "attention_mask", "labels", "token_type_ids"]:
                    if key not in output_batch:
                        continue
                    output_batch[key] = output_batch[key][:, seq_start:]
        return output_batch