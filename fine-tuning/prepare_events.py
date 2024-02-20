import os
import json



def prepare_events():
    data1 = json.load(open("../data/original/test.json"))
    events = []
    for item in data1:
        events.append(item["event"])
    print(len(events))

    with open("events.txt", "w") as f:
        for event in events:
            f.write(event+"\n")

if __name__ == "__main__":
    prepare_events()