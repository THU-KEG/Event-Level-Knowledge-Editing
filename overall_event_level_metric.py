import os
import json
from collections import defaultdict


def parse_answer(text):
    # return text.strip().lower()
    # return text.split("\n")[0].strip().lower()
    text = text.split("\n")[0].strip().lower()
    if text != "" and text[-1] == ".":
        return text[:-1]
    return text


def compute_fact(item):
    pred = item["request"]["result"]["completions"][0]["text"]
    labels = item["instance"]["answer"]["alias"] + [item["instance"]["answer"]["name"]]
    correct = 0
    # unknown
    if "unknown" in labels:
        if "unknown" in parse_answer(pred):
            correct = 1
    else:
        if parse_answer(pred) in [parse_answer(_label) for _label in labels]:
            correct = 1
    return correct


def get_all_event_level(fact_path, tendency_path):
    fact = json.load(open(fact_path))
    tendency = json.load(open(tendency_path))
    event2fact = defaultdict(list)
    for item in fact["request_states"]:
        if item["instance"]["local"]: continue
        event = item["instance"]["event"]
        event2fact[event].append(compute_fact(item))
    event2tendency = defaultdict(list)
    for item in tendency["request_states"]:
        if item["instance"]["local"]: continue
        event = item["instance"]["event"]
        correct = int(item["scores"]["overall"] == 5)
        event2tendency[event].append(correct)
    in_fact = 0
    for event in event2tendency:
        if event in event2fact:
            in_fact += 1
            for score in event2fact[event]:
                event2tendency[event].append(score)
    # print(in_fact)
    correct = 0
    for event in event2tendency:
        flag = 1
        for score in event2tendency[event]:
            flag *= score
        correct += flag
    print(correct, len(event2tendency), correct/len(event2tendency)*100)


if __name__ == "__main__":
    models = ["gpt-3.5", "gpt-4", "gemini-pro"]
    files = [
        ("fact_bm25.json", "tendency_gen_bm25_exam.json"),
        ("fact_e5.json", "tendency_gen_e5_exam.json"),
        ("fact_serac.json", "tendency_gen_serac_exam.json"),
        ("fact.json", "tendency_gen_exam.json")
    ]
    for model in models:
        for file in files:
            fact_path = f"data/processed/fact/{model}/{file[0]}"
            tendency_path = f"data/processed/tendency/{model}/examiner/{file[1]}"
            get_all_event_level(fact_path, tendency_path)


    models = ["gpt-j-6b", "tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    files = [
        ("fact_ft_predictions.json", "tendency_gen_ft_exam.json"),
        ("fact_bm25_predictions.json", "tendency_gen_bm25_exam.json"),
        ("fact_e5_predictions.json", "tendency_gen_e5_exam.json"),
        ("fact_serac_predictions.json", "tendency_gen_serac_exam.json"),
        ("fact_predictions.json", "tendency_gen_exam.json")
    ]
    for model in models:
        for file in files:
            fact_path = f"open-source/output/{model}/{file[0]}"
            tendency_path = f"open-source/output/{model}/examiner/{file[1]}"
            get_all_event_level(fact_path, tendency_path)