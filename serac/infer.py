import os
from pathlib import Path
import sys
sys.path.append("../../")
import json
import logging
import numpy as np

from transformers import set_seed, Trainer

from dataloader import InferClassifierDataset, collate_fn
from model import get_model
from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) >= 2 and sys.argv[-1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# output dir
model_name_or_path = model_args.model_name_or_path.split("/")[-1]

# logging config
logging.basicConfig(
    filename=os.path.join(training_args.output_dir, "running.log"),
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# logging
logging.info(data_args)
logging.info(model_args)
logging.info(training_args)

# set seed
set_seed(training_args.seed)

# model
tokenizer, model = get_model(model_args)
model.cuda()

# metric
def comput_acc(prediction):
    logits, labels = prediction[0], prediction[1]
    preds = np.int32(logits > 0.5)
    correct = np.sum((preds == labels) * (labels == 1))
    ground_truth = np.sum(labels)
    num_preds = np.sum(preds)
    precision = correct / (num_preds + 1e-10)
    recall = correct / ground_truth
    f1 = 2*precision*recall / (precision+recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Trainer 
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics=comput_acc,
    data_collator=collate_fn,
    tokenizer=tokenizer
)

logging.info("Evaluating...")
test_dataset = InferClassifierDataset(data_args, tokenizer, data_args.test_file)
logits, labels, metrics = trainer.predict(test_dataset=test_dataset, ignore_keys=["loss"])
logging.info("Results: {}".format(metrics))
# import pdb; pdb.set_trace()
print(metrics)
np.save(os.path.join(training_args.output_dir, "results.npy"), logits)