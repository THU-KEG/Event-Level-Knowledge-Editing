import json

def load_fact_corpus(data_path):
    event_corpus = []
    with open(data_path) as f:
        data = json.load(f)
        for item in data:
            event_corpus.append(item["event"])
    return event_corpus


def load_tendency_corpus(data_path):
    event_corpus = []
    with open(data_path) as f:
        data = json.load(f)
        for item in data:
            event_corpus.append(item["event"])
    return event_corpus