import os
import re
import json
import random
from collections import defaultdict, Counter

def parse_answer(text):
    # return text.strip().lower()
    # return text.split("\n")[0].strip().lower()
    text = text.split("\n")[0].strip().lower()
    if text != "" and text[-1] == ".":
        return text[:-1]
    return text

class MetricForFact():
    def __init__(self, predictions, labels, ids, data) -> None:
        self.predictions = predictions
        self.labels = labels
        self.ids = ids
        self.data = data
        self.metrics = {}

    @staticmethod
    def load_file(answer_file):
        predictions, labels = [], []
        ids = []
        data = json.load(open(answer_file))
        filtered_num = 0
        for i, item in enumerate(data["request_states"]):
            predictions.append(item["request"]["result"]["completions"][0]["text"])
            labels.append(item["instance"]["answer"]["alias"] + [item["instance"]["answer"]["name"]])
            if item["instance"]["answer"]["id"] == "Q30":
                labels[-1].append("United States")
            ids.append(
                (item["instance"]["item_id"], item["instance"]["qa_id"], i)
            )
        print(filtered_num)
        return data, predictions, labels, ids


    def compute_acc(self):
        counter = Counter()
        # overall
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if parse_answer(pred) in [parse_answer(_label) for _label in label]:
                counter["correct"] += 1
            else:
                pass
            # overall
            counter["total"] += 1
        
        counter["acc"] = counter["correct"] / counter["total"]
        self.metrics["acc"] = counter
        print(counter)


    def dump_metrics(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)


if __name__ == "__main__":
    # models = ["gpt-4", "gpt-3.5", "gemini-pro"]
    # models = ["gpt-3.5", "gpt-4", "gemini-pro"]
    models = ["gpt-j-6b", "tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    files = [
        ("fact_recall.json", "metrics.json"),
    ]
    for model in models:
        for file in files:
            # io_dir = f"data/{model}"
            io_dir = f"../open-source/output/{model}"
            answer_file = os.path.join(io_dir, "fact_recall_predictions.json")
            data, predictions, labels, ids = MetricForFact.load_file(answer_file)
            
            metric = MetricForFact(predictions, labels, ids, data)
            metric.compute_acc()
            metric.dump_metrics(os.path.join(io_dir, file[1]))


