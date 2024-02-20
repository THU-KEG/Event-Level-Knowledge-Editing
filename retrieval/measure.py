import os
import json
from collections import Counter


def measure_retrieval_acc(data_path, q_type, dump_dir=None):
    data = []
    with open(data_path) as f:
        data = json.load(f)
    correct = 0
    total = 0
    incorrect = []
    for item in data:
        for qa in item[q_type]["qas"]:
            total += 1
            if qa["retrieved_event"] == item["event"]:
                correct += 1
            else:
                if (item["event"], qa["retrieved_event"]) not in incorrect:
                    incorrect.append((item["event"], qa["retrieved_event"]))
        # for qa in item[q_type]["local_qas"]:
        #     total += 1
        #     if qa["retrieved_event"] == item["event"]:
        #         correct += 1
        #     else:
        #         if (item["event"], qa["retrieved_event"]) not in incorrect:
        #             incorrect.append((item["event"], qa["retrieved_event"], qa["question"]))
    print(correct, total, correct/total)
    if dump_dir is not None:
        json.dump(incorrect, open(os.path.join(dump_dir, "retrieval_errors.json"), "w"), indent=4)


if __name__ == "__main__":
    retriever = "multilingual-e5-large"
    # retriever = "bm25"
    # type = "fact"
    type = "tendency"
    dump_dir = f"data/{retriever}/{type}"
    measure_retrieval_acc(f"data/{retriever}/{type}/test-with-retrieval.json", type, dump_dir)
