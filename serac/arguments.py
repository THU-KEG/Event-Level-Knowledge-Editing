import os
import yaml 
import json
import dataclasses

from enum import Enum
from pathlib import Path 
from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.

    Arguments pertaining to what data we are going to input our model for training and eval, such as the config file
    path, dataset name, and the path of the training, validation, and testing file. By using `HfArgumentParser`, we can
    turn this class into argparse arguments to be able to specify them on the command line.
    """
    # output_dir: str = field(
    #     metadata={"help": "output dir"}
    # )
    config_file: str = field(
        default=None, 
        metadata={"help": "Config file path."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the test data."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={
            "help": "Whether return token type ids"
        }
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from, such as the model type, model
    path, checkpoint path, hidden size, and aggregation method.
    """
    model_type: str = field(
        default="bert",
        metadata={"help": "Model type."}
    )
    model_name_or_path: str = field(
        default="/data3/MODELS/distilbert-base-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)


@dataclass 
class TrainingArguments(TrainingArguments):
    """Arguments pertaining to the configurations in the training process.

    Arguments pertaining to the configurations in the training process, such as the random seed, task name,
    early stopping patience and threshold, and max length.
    """
    seed: int = field(
        default=42,
        metadata={"help": "seed"}
    )


class ArgumentParser(HfArgumentParser):
    """Alternative helper method that does not use `argparse` at all.

    Alternative helper method that does not use `argparse` at all, parsing the pre-defined yaml file with arguments
    instead loading a json file and populating the dataclass types.
    """
    def parse_yaml_file(self, yaml_file: str):
        """Parses the pre-defined yaml file with arguments."""
        data = yaml.safe_load(Path(yaml_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)