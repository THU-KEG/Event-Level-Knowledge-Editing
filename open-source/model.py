import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    AutoTokenizer, LlamaTokenizer
)
import logging
from arguments import ModelArguments

def get_backbone(model_type: str,
                 model_name_or_path: str,
                 tokenizer_name: str,
                 markers: List[str],
                 model_args: Optional[ModelArguments] = None,
                 new_tokens: Optional[List[str]] = []):
    if model_type == "Seq2Seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "CausalLM":
        logging.info("CausalLM: from {} load".format(model_name_or_path))
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, never_split=markers, padding_side="left")
        # tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, never_split=markers)
    else:
        raise ValueError("Invalid parameters `model_type`: %s" % model_type)

    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens=True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    config = model.config
    if model_type == "CausalLM":
        config.pad_token_id = config.eos_token_id
    return model, tokenizer, config
