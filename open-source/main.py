import os
import re
import sys
from pathlib import Path
import json
import logging
import numpy  as np
from collections import Counter

from transformers import set_seed
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer

from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from model import get_backbone
from data_loader import Seq2SeqProcessor, CausalLMProcessor, CausalLMFactProcessor

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) >= 2 and sys.argv[-1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# output dir
# model_name_or_path = model_args.model_name_or_path.split("/")[-1]
model_name_or_path = model_args.model_name_or_path
output_dir = Path(training_args.output_dir, f"{model_name_or_path.split('/')[-1]}")
output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = str(output_dir)

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# sync config
data_args.generation_max_length = training_args.generation_max_length

# markers
markers = ["<entity>", "</entity>", "<event>", "</event>", "<head>", "</head>", "<tail>", "</tail>"]

# logging
logging.info(data_args)
logging.info(model_args)
logging.info(training_args)

# set seed
set_seed(training_args.seed)

# writter 
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
model, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, markers,
                                           new_tokens=markers)
if model_args.model_type == "Seq2Seq":
    data_class = Seq2SeqProcessor
elif model_args.model_type == "CausalLM":
    # data_class = CausalLMProcessor
    data_class = CausalLMFactProcessor
else:
    raise ValueError

# metric
def parse_triples(text):
    pattern = re.compile("\((.*);(.*);(.*)\)")
    triples = []
    for span in text.split("|"):
        triples.extend(re.findall(pattern, span))
    if len(triples) == 0:
        return None
    for i, triple in enumerate(triples):
        triples[i] = (triple[0].strip(), triple[1].strip(), triple[2].strip())
    if "maven-ere" in data_args.dataset_name or data_args.dataset_name == "matres":
        final_triples = []
        for triple in list(set(triples)):
            ts = triple[2].split(",")
            for t in ts:
                final_triples.append((triple[0], triple[1], t.strip()))
        final_triples = list(set(final_triples))
    else:
        final_triples = list(set(triples))
    return final_triples

def compute_per_cls(metric_per_class, gold_triples, pred_triples):
    # record per class
    for triple in gold_triples:
        if triple[-1] not in metric_per_class:
            metric_per_class[triple[-1]] = Counter()
        metric_per_class[triple[-1]]["n_gold"] += 1
    for triple in pred_triples:
        if triple[-1] not in metric_per_class:
            metric_per_class[triple[-1]] = Counter()
        metric_per_class[triple[-1]]["n_pred"] += 1
    for triple in gold_triples:
        if triple in pred_triples:
            metric_per_class[triple[-1]]["tp"] += 1
    return metric_per_class

def comput_f1(prediction):
    logits, labels = prediction[0], prediction[1]
    logits = np.where(logits != -100, logits, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)

    # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    decoded_labels = data_args.all_labels

    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    
    tp = 0
    n_gold = 0
    n_pred = 0
    metric_per_class = dict()
    for pred_text, gold_text in zip(decoded_preds, decoded_labels):
        pred_text = clean_str(pred_text)
        gold_text = clean_str(gold_text)
        # gold triple
        gold_triples = []
        if gold_text != "NA" and gold_text != "":
            gold_text = gold_text.lower()
            triples = parse_triples(gold_text)
            if triples is not None:
                gold_triples.extend(triples)
        # pred triple
        pred_triples = []
        if pred_text != "NA" and pred_text != "":
            pred_text = pred_text.lower()
            triples = parse_triples(pred_text)
            if triples is not None:
                pred_triples.extend(triples)
        for triple in gold_triples:
            if triple in pred_triples:
                tp += 1
        n_gold += len(gold_triples)
        n_pred += len(pred_triples)
        metric_per_class = compute_per_cls(metric_per_class, gold_triples, pred_triples)
    # micro f1
    if tp != 0:
        precision = tp / n_pred
        recall = tp / n_gold
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0
    micro_metric = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    # macro f1
    for _cls in metric_per_class:
        precision = metric_per_class[_cls]["tp"] / (metric_per_class[_cls]["n_pred"]+1e-10)
        recall = metric_per_class[_cls]["tp"] / (metric_per_class[_cls]["n_gold"]+1e-10)
        metric_per_class[_cls]["precision"] = precision
        metric_per_class[_cls]["recall"] = recall
        metric_per_class[_cls]["f1"] = 2 * precision * recall / (precision + recall + 1e-10)
    macro_metric = dict()
    for key in ["precision", "recall", "f1"]:
        macro_metric[key] = 0
        for _cls in metric_per_class.keys():
            macro_metric[key] += metric_per_class[_cls][key]
        macro_metric[key] /= len(metric_per_class)
    # return 
    if data_args.dataset_name == "goemo":
        return macro_metric
    return micro_metric

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file, True)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file, False)
# data_args.all_labels = eval_dataset.get_labels()

# Trainer
if model_args.model_type == "Seq2Seq":
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=comput_f1,
        data_collator=train_dataset.collate_fn,
        tokenizer=tokenizer,
        # callbacks=[earlystoppingCallBack],
    )

    if training_args.do_train:
        logging.info("Training...")
        trainer.train()
        if training_args.save_at_end:
            trainer.save_model()

    if training_args.do_predict:
        logging.info("Evaluating...")
        test_dataset = data_class(data_args, tokenizer, data_args.test_file, False)
        # used for testing
        data_args.all_labels = test_dataset.get_labels()
        logits, labels, metrics = trainer.predict(test_dataset=test_dataset, ignore_keys=["loss"])
        logging.info("Results: {}".format(metrics))
        json.dump(metrics, open(os.path.join(output_dir, "results.json"), "w"), indent=4)
        
        # dump result
        logits = np.where(logits != -100, logits, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = data_args.all_labels

        if training_args.local_rank in [0, -1]:
            with open(os.path.join(output_dir, "predictions.json"), "w") as f:
                data = []
                for pred, label in zip(decoded_preds, decoded_labels):
                    data.append({
                        "pred": pred,
                        "label": label
                    })
                json.dump(data, f, indent=4)

# lcy 20240126
elif model_args.model_type == "CausalLM":
    if training_args.do_train:
        print("Training...")
        #TODO
    
    if training_args.do_predict:
        print("Evaluating")
        test_dataset = data_class(data_args, tokenizer, data_args.test_file, False)
        # used for testing
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            output_json_file = json.load(f)
        completions = []
        for i, example in enumerate(test_dataset.input_features):
            if i % 100 == 0:
                print(i)
            input_ids = example.input_ids
            outputs = model.generate(input_ids=input_ids, max_length=training_args.generation_max_length)
            completions.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        # dump result
        for i in range(len(output_json_file["request_states"])):
            output_json_file["request_states"][i]["request"]["result"]["completions"][0]["text"] = completions[i]
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(output_json_file, f, indent=4)
        

else:
    raise ValueError("Invalid parameters `model_type`: %s" % model_args.model_type)