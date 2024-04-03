import os
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from prompt import local_examples
import argparse


def load_generation(input_path, local_path):
    _data = json.load(open(input_path))
    local_data = json.load(open(local_path))
    data = []
    for item in _data["request_states"]:
        if item["instance"]["local"]: data.append(item)
    final_data = []
    for item1, item2 in zip(data, local_data["request_states"]):
        assert item1["instance"]["event"] == item2["instance"]["event"]
        assert item1["instance"]["input"]["text"].split("\n")[-2] == item2["instance"]["input"]["text"].split("\n")[-2]
        final_data.append({
            "answer1": item1["request"]["result"]["completions"][0]["text"],
            "answer2": item2["request"]["result"]["completions"][0]["text"]
        })
    return final_data


def construct_query(input_path, local_path, save_path):
    instruction = "You are a fair assessment expert, and please decide whether the two answers below are consistent. "
    instruction += "Your task is to score the consistency of the two answers according to the following requirements: "
    instruction += "1. You should assign a consistency score between 1 and 5, with 5 being the highest. "
    instruction += "2. Please score primarily on the basis of the tendency judgments in the two answers, and score 1 if the tendencies mentioned in two answers are completely inconsistent. "
    instruction += "Please output the score directly, like Score: 3.\n"
    instruction += "Please do not answer with any text, just output the score."
    import pdb; pdb.set_trace()
    demonstrations = []
    for example in local_examples:
        demonstrations.append({
            "input": f"Answer1: {example['answer1']} Answer2: {example['answer2']}\n",
            "output": f"Score: {example['label']}\n"
        })

    data = load_generation(input_path, local_path)
    print(len(data))
    group_size = 1
    grouped_data = []
    num_groups = len(data) // group_size if len(data) % group_size == 0 else len(data) // group_size + 1
    for i in range(num_groups):
        grouped_data.append(data[i*group_size: (i+1)*group_size])

    data = {
        "prompt": {
            "instructions": instruction,
            "input_prefix": "",
            "input_suffix": "\n",
            "output_prefix": "",
            "output_suffix": "\n",
            "demonstrations": demonstrations
        },
        "request_states": [
        ]
    }
    idx = 0
    for items in grouped_data:
        input = ""
        numbers = ["I", "II", "III", "IV", "V"]
        for i, item in enumerate(items):
            # input += "{}. {}\n".format(numbers[i], f"Answer1: {item['answer1']} Answer2: {item['answer2']}")
            input += "{}".format(f"Answer1: {item['answer1']} Answer2: {item['answer2']}")

        instance =  {
            "instance": {
                "input": {
                    "text": input
                },
                "id": idx,
            },
            "request": {
                "result": {
                    "success": False,
                    "completions": [
                        {
                            "text": ""
                        }
                    ]
                },
                "request_time": 0,
                "request_datetime": 0
            }
        }
        data["request_states"].append(instance)
        idx += 1
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query data for locality")
    parser.add_argument("--model", type=str, default="gpt-3.5")
    parser.add_argument("--file", type=str, default="gen", options=["gen_ft", "gen_bm25", "gen_e5", "gen_serac", "gen"])
    args = parser.parse_args()


    if args.model in ["gpt-j-6b", "Mistral-7B-Instruct-v0.2", "tulu-v2-7b"]:
        io_dir = f"../open-source/output/{args.model}"
        save_dir = Path(os.path.join(io_dir, "examiner-local"))
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"tendency_{args.file}_predictions.json"
        input_path = os.path.join(io_dir, file_name)
        local_path = os.path.join(io_dir, f"tendency_gen_local_predictions.json")
        save_path = os.path.join(save_dir, file_name)

        construct_query(input_path, local_path, save_path)
    else:
        io_dir = f"../data/processed/tendency/{args.model}"
        save_dir = Path(os.path.join(io_dir, "examiner-local"))
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"tendency_{args.file}.json"
        input_path = os.path.join(io_dir, file_name)
        local_path = os.path.join(io_dir, f"tendency_gen_local.json")
        save_path = os.path.join(save_dir, file_name)

        construct_query(input_path, local_path, save_path)
