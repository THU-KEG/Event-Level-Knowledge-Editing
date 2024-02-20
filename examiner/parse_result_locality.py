import os
import json
from collections import Counter


def parse_score(score):
    score = int(score.split(":")[-1])
    return score


def parse_result(input_path):
    _data = json.load(open(input_path))
    data = []
    for item in _data["request_states"]:
        output = item["request"]["result"]["completions"][0]["text"].strip()
        scores = output.split("\n")
        # import pdb; pdb.set_trace()
        for score in scores[:1]:
            try:
                data.append({
                    "consistency": parse_score(score)
                })
            except:
                print(score)
                data.append({
                    "consistency": -1
                })

    total = 0
    consistent = 0
    full_mark = 0
    # import pdb; pdb.set_trace()
    for item in data:
        if item["consistency"] == -1:
            continue
        total += 1
        consistent += item["consistency"]
        if item["consistency"] == 5:
            full_mark += 1
    # print(consistent, total, consistent / total)
    print(full_mark, total, full_mark/total * 100)


if __name__ == "__main__":
    # files = [
    #     "tendency_gen_bm25.json", 
    #     "tendency_gen_e5.json", 
    #     "tendency_gen_serac.json", 
    #     "tendency_gen.json"
    # ]
    # model = "gpt-3.5"
    # for file in files:
    #     input_path = f"../data/processed/tendency/{model}/examiner-local/{file}"
        # parse_result(input_path)
    
    # models = ["tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    models = ["gpt-j-6b"]
    files = ["gen_ft", "gen_bm25", "gen_e5", "gen_serac", "gen"]
    for model in models:
        for file in files:
            input_path = f"../open-source/output/{model}/examiner-local/tendency_{file}_predictions.json"
            parse_result(input_path)
