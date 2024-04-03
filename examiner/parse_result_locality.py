import os
import json
from collections import Counter
import argparse


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
    parser = argparse.ArgumentParser(description="Parse for locality")
    parser.add_argument("--model", type=str, default="gpt-3.5")
    parser.add_argument("--file", type=str, default="gen", options=["gen_ft", "gen_bm25", "gen_e5", "gen_serac", "gen"])
    args = parser.parse_args()

    if args.model in ["gpt-j-6b", "tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]:
        input_path = f"../open-source/output/{args.model}/examiner-local/tendency_{args.file}_predictions.json"
        parse_result(input_path)
    else:
        input_path = f"../data/processed/tendency/{args.model}/examiner-local/tendency_{args.file}.json"
    parse_result(input_path)