import os
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from prompt import examples


def load_generation(input_path):
    _data = json.load(open(input_path))
    data = []
    for item in _data["request_states"]:
        if "\n\n" not in item["instance"]["input"]["text"]:
            if len(item["instance"]["input"]["text"].split("\n")) == 3:
                event, question, _ = item["instance"]["input"]["text"].split("\n")
            else: # ft
                event = "NA"
                question, _  = item["instance"]["input"]["text"].split("\n")
        else:
            input = item["instance"]["input"]["text"].split("\n\n")[-1]
            if len(input.split("\n")) == 3:
                event, question, _ = input.split("\n")
            else: # serac
                event = "NA"
                question, _ = input.split("\n")
        if item["instance"]["local"]: continue
        data.append({
            "event": event,
            "question": question,
            "answer": item["request"]["result"]["completions"][0]["text"],
            "label": item["instance"]["answer"]
        })
    return data


def construct_query(input_path, save_path):
    instruction = "You are a fair assessment expert, and you will be given a set of (event, question, answer, label) quadruples. "
    instruction += "The answer is obtained by answering the question based on the event (assuming the event has occurred). "
    instruction += "The answer usually consists of a judgment (or a claim) and some reasons (or explanations) for the judgment. "
    instruction += "The label is a general judgment annotated by humans for the question. "
    instruction += "Your task is to score the answers according to the following requirements:\n"
    instruction += "a. You should score the answer based on the provided label.\n"
    instruction += "b. You should rate the answer on 4 metrics, for the first 3 metrics, assign a score between 1 and 5, with 5 being the highest: "
    instruction += "1. For accuracy, you will score whether the answer correctly answers the question, i.e., whether the judgment in the answer is consistent with the provided label. "
    instruction += "2. For coherence, you will assess whether the reasons and explanations in the answer are reasonable and logically consistent. "
    instruction += "3. For comprehensive, you will determine if the answer covers multiple aspects of the reasons and explanations and provides a comprehensive response. "
    instruction += "4. Finally, you will provide an overall score between 1 and 5, with 5 being the highest.\n"
    instruction += "If accuracy <= 3, the overall should not be higher than 3.\n"
    instruction += "You should only give the integer score.\n"
    instruction += "DO NOT complete the answer!\n"
    instruction += "In the input query we identify each one with a Roman numeral, please quadruples the corresponding Roman numeral and its score in the output."
    import pdb; pdb.set_trace()
    demonstrations = []
    for example in examples:
        demonstrations.append({
            "input": f"Event: {example['event']} Question: {example['question']} Answer: {example['answer']} Label: {example['label']}\n",
            "output": f"Accuracy: {example['score']['accuracy']}; Coherence: {example['score']['coherence']}; Comprehensive: {example['score']['comprehensive']}; Overall: {example['score']['overall']}\n"
        })

    data = load_generation(input_path)
    print(len(data))
    # exit()
    group_size = 5
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
            input += "{}. {}\n".format(numbers[i], f"{item['event']} {item['question']} Answer: {item['answer']} Label: {item['label']}")
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
    # io_dir = "../data/processed/tendency/gemini-pro"
    # # io_dir = "../open-source/output512/Mistral-7B-Instruct-v0.2"
    # save_dir = Path(os.path.join(io_dir, "examiner"))
    # save_dir.mkdir(parents=True, exist_ok=True)

    # file_name = "tendency_gen_serac.json"
    # input_path = os.path.join(io_dir, file_name)
    # save_path = os.path.join(save_dir, file_name)

    # construct_query(input_path, save_path)
    
    for model in ["gpt-j-6b", "Mistral-7B-Instruct-v0.2", "tulu-v2-7b"]:
        for file_name in ["tendency_gen_predictions.json", "tendency_gen_serac_predictions.json", "tendency_gen_ft_predictions.json", "tendency_gen_bm25_predictions.json", "tendency_gen_e5_predictions.json"]:
            # io_dir = f"../data/processed/tendency/{model}"
            io_dir = f"../open-source/output/{model}"
            save_dir = Path(os.path.join(io_dir, "examiner"))
            save_dir.mkdir(parents=True, exist_ok=True)

            input_path = os.path.join(io_dir, file_name)
            save_path = os.path.join(save_dir, file_name)

            construct_query(input_path, save_path)


