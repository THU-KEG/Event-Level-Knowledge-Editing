import os
import json
from collections import Counter


def parse_score(score):
    metric = {}
    for score in score.split(".")[-1].strip().split(";"):
        name, number = score.split(":")
        metric[name.strip().lower()] = int(number.strip())
    return metric


def parse_result(input_path, pred_path, dump_path=None):
    _data = json.load(open(input_path))
    data = []
    for i, item in enumerate(_data["request_states"]):
        output = item["request"]["result"]["completions"][0]["text"].strip()
        scores = output.split("\n")
        for score in scores[:5]:
            try:
                data.append({
                    "scores": parse_score(score)
                })
            except:
                print(score)
                data.append({
                    "scores": {
                        "accuracy": -1,
                        "coherence": -1,
                        "comprehensive": -1,
                        "overall": -1
                    }
                })
        if len(scores) < 5 and i != len(_data["request_states"]) - 1:
            for _ in range(5-len(scores)):
                data.append({
                    "scores": {
                        "accuracy": -1,
                        "coherence": -1,
                        "comprehensive": -1,
                        "overall": -1
                    }
                })

    pred_data = json.load(open(pred_path))
    # pred_data["request_states"] = [item for item in pred_data["request_states"] if item["request"]["result"]["success"]]
    # data = data[:len(pred_data["request_states"])]
    # import pdb; pdb.set_trace()
    # assert len(data) == len(pred_data["request_states"])
    final_data = {
        "prompt": pred_data["prompt"],
        "request_states": []
    }
    idx = 0
    # import pdb; pdb.set_trace()
    for item in pred_data["request_states"]:
        if item["instance"]["local"]: continue
        if idx == len(data):
            break
        item["scores"] = data[idx]["scores"]
        final_data["request_states"].append(item)
        idx += 1
    # import pdb; pdb.set_trace()
    # assert len(final_data["request_states"]) == len(data)
    
    if dump_path is None:
        with open(pred_path, "w") as f:
            json.dump(final_data, f, indent=4)
    else:
        with open(dump_path, "w") as f:
            json.dump(final_data, f, indent=4)
    

def compute_score(path):
    data = json.load(open(path))
    counter = Counter()
    total = 0
    for item in data["request_states"]:
        if item["instance"]["local"]: continue
        if item["scores"]["overall"] == -1: continue
        total += 1
        for key in item["scores"]:
            counter[key] += item["scores"][key]
    for key in counter:
        counter[key] /= total
    print(counter)


if __name__ == "__main__":
    # model = "gemini-pro"
    # input_path = f"../data/processed/tendency/{model}/examiner/tendency_gen_bm25.json"
    # pred_path = f"../data/processed/tendency/{model}/tendency_gen_bm25.json"
    # dump_path = f"../data/processed/tendency/{model}/examiner/tendency_gen_bm25_exam.json"
    # models = ["gpt-j-6b", "tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    models = ["gpt-j-6b"]
    files = ["gen_ft", "gen_bm25", "gen_e5", "gen_serac", "gen"]
    for model in models:
        for file in files:
            input_path = f"../open-source/output/{model}/examiner/tendency_{file}_predictions.json"
            pred_path = f"../open-source/output/{model}/tendency_{file}_predictions.json"
            dump_path = f"../open-source/output/{model}/examiner/tendency_{file}_exam.json"
            print(input_path)
            parse_result(input_path, pred_path, dump_path)
            compute_score(dump_path)