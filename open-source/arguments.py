import os
import yaml 
import json
import dataclasses

from enum import Enum
from pathlib import Path 
from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import Seq2SeqTrainingArguments, HfArgumentParser


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.

    Arguments pertaining to what data we are going to input our model for training and eval, such as the config file
    path, dataset name, and the path of the training, validation, and testing file. By using `HfArgumentParser`, we can
    turn this class into argparse arguments to be able to specify them on the command line.
    """
    config_file: str = field(
        default=None, 
        metadata={"help": "Config file path."}
    )
    examples_proportional_k: Optional[int] = field(
        default=10000,
        metadata={"help": "K in Examples-proportional mixing"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
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
    train_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers for training data. (Only meaningful for EAE)"}
    )
    validation_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers for valid data. (Only meaningful for EAE)"}
    )
    test_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers test data. (Only meaningful for EAE)"}
    )
    test_exists_labels: bool = field(
        default=False,
        metadata={"help": "Whether test dataset exists labels"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_out_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total output sequence length after tokenization."
        }
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={
            "help": "Whether return token type ids"
        }
    )
    truncate_seq2seq_output: bool = field(
        default=False,
        metadata={
            "help": "Used for Seq2Seq. Whether truncate output labels."
        }
    )
    truncate_in_batch: bool = field(
        default=False,
        metadata={
            "help": "whether truncate in batch. False only if mrc."
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
        metadata={
            "help": "Model type.",
            "choices": ["Seq2Seq", "CausalLM"]
        },
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    backbone_checkpoint_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_checkpoint_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier"}
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
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
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
class TrainingArguments(Seq2SeqTrainingArguments):
    """Arguments pertaining to the configurations in the training process.

    Arguments pertaining to the configurations in the training process, such as the random seed, task name,
    early stopping patience and threshold, and max length.
    """
    seed: int = field(
        default=42,
        metadata={"help": "seed"}
    )
    save_at_end: bool = field(
        default=False,
        metadata={"help": "Whether save at the end of training."}
    )
    early_stopping_patience: int = field(
        default=7,
        metadata={"help": "Patience for early stopping."}
    )
    early_stopping_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold for early stopping."}
    )
    generation_max_length: int = field(
        default=128, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    generation_num_beams: int = field(
        default=3, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    ignore_pad_token_for_loss: bool = field(
        default=False, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    predict_with_generate: bool = field(
        default=False, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
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