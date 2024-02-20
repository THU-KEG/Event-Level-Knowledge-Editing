import os
import json
import numpy as np
from collections import Counter

def retrieve_event(test_file, match_scores, type, save_path, threshold=0.5):
    data = json.load(open(test_file))
    events = []
    num_qs = 0
    for item in data:
        events.append(item["event"])
        num_qs += len(item[type]["qas"])
        num_qs += len(item[type]["local_qas"])
    assert len(events) * num_qs == match_scores.shape[0]
    q_idx = 0

    counter = Counter()
    for item in data:
        for qa in item[type]["qas"]:
            q_scores = match_scores[q_idx*len(events):(q_idx+1)*len(events)]
            e_idx = np.argmax(q_scores)
            q_idx += 1
            if q_scores[e_idx] >= threshold:
                event = events[e_idx]
                qa["serac_event"] = event
            else:
                qa["serac_event"] = "NA"
            if qa["serac_event"] == item["event"]:
                counter["correct"] += 1

        for qa in item[type]["local_qas"]:
            q_scores = match_scores[q_idx*len(events):(q_idx+1)*len(events)]
            e_idx = np.argmax(q_scores)
            q_idx += 1
            if q_scores[e_idx] >= threshold:
                event = events[e_idx]
                qa["serac_event"] = event
            else:
                qa["serac_event"] = "NA"
            if qa["serac_event"] == "NA":
                counter["correct"] += 1
    print(counter["correct"], num_qs, counter["correct"]/num_qs)
    
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    type = "tendency"
    test_file = f"../data/original/test.json"
    match_scores = np.load(f"output/{type}/results.npy")
    save_path = f"output/{type}/test-with-serac.json"
    retrieve_event(test_file, match_scores, type, save_path)
