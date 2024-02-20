import re
import json
import logging
from typing import List, Union, Tuple, Optional

from tqdm import tqdm
from collections import defaultdict
from base_processor import (
    InputExample,
    InputFeatures,
    DataProcessor,
    # MultitaskDataProcessor
)


logger = logging.getLogger(__name__)


class Seq2SeqProcessor(DataProcessor):
    def __init__(self,
                 config,
                 tokenizer,
                 input_file: str,
                 is_train_set: bool=True) -> None:
        """Constructs a `Seq2SeqProcessor`."""
        super().__init__(config, tokenizer, is_train_set)
        print(self.config)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    

    def read_examples(self,
                      input_file: str) -> None:
        self.examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            if isinstance(all_data, list):
                for data in all_data:
                    for idx, item in enumerate(data["request_states"]):
                        # input_text = "[S2S] " + data["prompt"]["instructions"] + item["instance"]["input"]["text"] + " <extra_id_0>"
                        input_text = data["prompt"]["instructions"] + " " + item["instance"]["input"]["text"]
                        labels = item["instance"]["references"][0]["output"]["text"]
                        example = InputExample(
                            example_id = idx,
                            input = input_text,
                            labels = labels
                        )
                        self.examples.append(example)
            else:
                for idx, item in enumerate(all_data["request_states"]):
                    # input_text = "[S2S] " + all_data["prompt"]["instructions"] + item["instance"]["input"]["text"] + " <extra_id_0>"
                    input_text = all_data["prompt"]["instructions"] + "\n\nText: ``" + item["instance"]["input"]["text"] + "``.\nAnswer: "
                    # input_text = all_data["prompt"]["instructions"] + "\n" + item["instance"]["input"]["text"]
                    labels = item["instance"]["references"][0]["output"]["text"]
                    example = InputExample(
                        example_id = idx,
                        input = input_text,
                        labels = labels
                    )
                    self.examples.append(example)


    def convert_examples_to_features(self) -> None:
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for Seq2Seq"):
            # context 
            input_context = self.tokenizer(example.input,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=False)
            # output labels
            label_outputs = self.tokenizer(example.labels,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_out_length,
                                           is_split_into_words=False)
            # set -100 to unused token 
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = InputFeatures(
                example_id=example.example_id,
                input_ids=input_context["input_ids"],
                attention_mask=input_context["attention_mask"],
                labels=label_outputs["input_ids"],
            )
            self.input_features.append(features)


class CausalLMProcessor(DataProcessor):
    def __init__(self,
                 config,
                 tokenizer,
                 input_file: str, 
                 is_train_set: bool=True) -> None:
        """Constructs a `CausalLMProcessor`."""
        super().__init__(config, tokenizer, is_train_set)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    

    def read_examples(self,
                      input_file: str) -> None:
        self.examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data["request_states"]):
                # input_text = data["prompt"]["instructions"] + \
                #              data["prompt"]["input_prefix"] + \
                #              item["instance"]["input"]["text"] + \
                #              data["prompt"]["input_suffix"]
                # labels = data["prompt"]["output_prefix"] + \
                #          item["instance"]["references"][0]["output"]["text"] + \
                #          data["prompt"]["output_suffix"]
                input_text = item["instance"]["input"]["text"].lower()
                labels = item["instance"]["references"][0]["output"]["text"].lower()
                example = InputExample(
                    example_id = idx,
                    input = input_text,
                    labels = labels
                )
                self.examples.append(example)


    def convert_examples_to_features(self) -> None:
        self.input_features = []
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.warning("Setting `pad_token_id` to `eos_token_id: {}`".format(self.tokenizer.eos_token_id))
        for example in tqdm(self.examples, desc="Processing features for CausalLM"):
            if self.is_train_set: 
                # context
                input_context = self.tokenizer(example.input + " Answer: ",
                                            truncation=True,
                                            max_length=self.config.max_seq_length,
                                            is_split_into_words=False)
                # output labels
                label_outputs = self.tokenizer(example.labels,
                                            truncation=True,
                                            add_special_tokens=False,
                                            max_length=self.config.max_out_length-1,
                                            is_split_into_words=False)
            
                # add eos token for label_outputs
                label_outputs["input_ids"].append(self.tokenizer.eos_token_id)

                # concat input text and labels
                input_ids = input_context["input_ids"] + label_outputs["input_ids"]
                attention_mask = [1] * len(input_ids)
                labels = [-100] * len(input_context["input_ids"]) + label_outputs["input_ids"]

                # -- IMPORATION -- left padding
                if len(input_ids) < self.config.max_seq_length + self.config.max_out_length:
                    padding_num = self.config.max_seq_length + self.config.max_out_length - len(input_ids)
                    input_ids = [self.tokenizer.pad_token_id]*padding_num + input_ids
                    attention_mask = [0]*padding_num + attention_mask
                    labels = [-100]*padding_num + labels
                
                # import pdb; pdb.set_trace()
            else:
                # context
                input_context = self.tokenizer(example.input + " Answer: ",
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.config.max_seq_length,
                                            is_split_into_words=False)
                # output labels
                label_outputs = self.tokenizer(example.labels,
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.config.max_out_length,
                                            is_split_into_words=False)
                # truncation
                labels = label_outputs["input_ids"]
                input_ids = input_context["input_ids"]
                attention_mask = input_context["attention_mask"]

            features = InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            self.input_features.append(features)

# lcy 20240126
class CausalLMFactProcessor(DataProcessor):
    def __init__(self, config, tokenizer, input_file: str, is_train_set: bool=True) -> None:
        """Constructs a `CausalLMFactProcessor`."""
        super().__init__(config, tokenizer, is_train_set)
        self.read_examples(input_file)
        # self.convert_examples_to_features()

    def read_examples(self, input_file: str) -> None:
        self.examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data["request_states"]):
                input_text = data["prompt"]["instructions"] + "\n" + \
                             data["prompt"]["input_prefix"] + \
                             item["instance"]["input"]["text"]
                labels = item["instance"]["answer"]["name"]
                example = InputExample(
                    example_id = idx,
                    input = input_text,
                    labels = labels,
                    instruction = data["prompt"]["instructions"],
                    instance=item["instance"]
                )
                self.examples.append(example)
    
    def convert_examples_to_features(self) -> None:
        self.input_features = []
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.warning("Setting `pad_token_id` to `eos_token_id: {}`".format(self.tokenizer.eos_token_id))
        for example in tqdm(self.examples, desc="Processing features for CausalLM"):
            if self.is_train_set:
                #TODO
                input_context = self.tokenizer(example.input + " Answer: ",
                                            truncation=True,
                                            max_length=self.config.max_seq_length,
                                            is_split_into_words=False)
                input_ids = input_context["input_ids"]
            else:
                input_context = self.tokenizer(example.input, truncation=True, padding="max_length", max_length=self.config.max_seq_length, is_split_into_words=False, return_tensors="pt")
                # label_outputs = self.tokenizer(example.labels, truncation=True, padding="max_length", max_length=self.config.max_out_length, is_split_into_words=False)
                input_ids = input_context["input_ids"]
                # attention_mask = input_context["attention_mask"]

            features = InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=None,
                labels=None
            )
            self.input_features.append(features)


