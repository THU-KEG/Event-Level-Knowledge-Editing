
from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
import json
import argparse
from tqdm import tqdm
import torch
import os
from pathlib import Path
from data_loader import Seq2SeqProcessor, CausalLMProcessor, CausalLMFactProcessor

models = ["/data3/MODELS/gpt-j-6b", "/data3/MODELS/Mistral-7B-Instruct-v0.2", "/data3/MODELS/tulu-v2-7b"]

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--max_out_length", type=int, default=512)
parser.add_argument("--return_token_type_ids", type=bool, default=False)
parser.add_argument("--truncate_in_batch", type=bool, default=True)
parser.add_argument("--truncate_seq2seq_output", type=bool, default=False)
parser.add_argument("--model_path", type=str, default="/data3/MODELS/gpt-j-6b")
parser.add_argument("--chat", action="store_true")


args = parser.parse_args()

DEBUG = False
CHAT = args.chat

model_path = args.model_path
print("model path: ", model_path)
if CHAT:
    print("chat mode")
outputpath = Path("./output", model_path.split("/")[-1])
# import pdb; pdb.set_trace()
outputpath.mkdir(exist_ok=True, parents=True)
outputpath = str(outputpath)
print("output path: ", outputpath)

if "gpt-j-6b" in model_path:
    args.max_out_length = 128

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# tokenizer.pad_token_id = tokenizer.eos_token_id
data_class = CausalLMFactProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("device: ", device)
# load dataset
data_root_path = "../data/processed/mistral-7b"
datasets = ["tendency_gen.json"]
for dataset in datasets:
    data_path = os.path.join(data_root_path, dataset)
    print("Evaluating on dataset: ", data_path)
    print("save to: ", os.path.join(outputpath, dataset.split(".")[0]+"_predictions.json"))
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    completions = []
    if DEBUG:
        data["request_states"] = data["request_states"][:5]

    for example in tqdm(data["request_states"]):
        # based on model type, we need to construct different inputs
        if CHAT:
            if "Mistral" in model_path:
                prompt = data["prompt"]["instructions"] + data["prompt"]["input_prefix"] + \
                                example["instance"]["input"]["text"]
                messages = [
                    {"role": "user", "content": prompt}
                ]
                encodeds = tokenizer.apply_chat_template(messages, max_length=args.max_seq_length, return_tensors="pt")
                inputs = encodeds.to(device)
            elif "tulu" in model_path:
                prompt = "<|user|>\n"
                prompt += data["prompt"]["instructions"]+data["prompt"]["input_prefix"] + example["instance"]["input"]["text"]
                prompt += "\n<|assistant|>\n"
                inputs = tokenizer(prompt, max_length=args.max_seq_length, return_tensors="pt")
                inputs = inputs.to(device)
                inputs = inputs.input_ids
            else:
                raise NotImplementedError
            generate_ids = model.generate(inputs, max_length=args.max_out_length, pad_token_id=tokenizer.eos_token_id)
            res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            res = res[len(prompt):]
            if "Mistral" in model_path:
                res = res.replace("Answer: [/INST]", "")
        else:
            prompt = data["prompt"]["instructions"] + data["prompt"]["input_prefix"] + example["instance"]["input"]["text"]
            encodeds = tokenizer(prompt, max_length=args.max_seq_length, return_tensors="pt")
            inputs = encodeds.to(device)
            generate_ids = model.generate(inputs.input_ids, max_length=args.max_out_length, attention_mask=inputs.attention_mask, pad_token_id=tokenizer.eos_token_id)
            res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            res = res[len(prompt):]
        res = res.strip()
        completions.append(res)
        if DEBUG:
            print("input: ", prompt)
            print("output: ", res)
            print()



    for i in range(len(completions)):
        data["request_states"][i]["request"]["result"]["completions"][0]["text"] = completions[i]

    if CHAT:
        with open(os.path.join(outputpath, dataset.split(".")[0]+"chat_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    else:
        with open(os.path.join(outputpath, dataset.split(".")[0]+"_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
