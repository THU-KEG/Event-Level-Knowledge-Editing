import os
import json
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
from rank_bm25 import BM25Okapi
from utils import load_fact_corpus, load_tendency_corpus


stopwords = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
with open("stopwords.txt") as f:
    for line in f.readlines():
        stopwords.append(line.strip())

class BM25Retriever():
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        self.tokenizer = BertTokenizer.from_pretrained("/data3/MODELS/bert-base-uncased")
        self.bm25 = self.index(corpus)

    def tokenize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        filtered_text = []
        for token in tokenized_text:
            if token in stopwords:
                continue
            filtered_text.append(token)
        return filtered_text

    def index(self, corpus):
        # import pdb; pdb.set_trace()
        tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def search(self, query, n=1):
        tokenized_query = self.tokenize(query)
        event = self.bm25.get_top_n(tokenized_query, self.corpus, n=n)[0]
        return event


def search_for_fact(data_path, bm25, save_dir):
    data = json.load(open(data_path))
    for item in tqdm(data):
        for qa in item["fact"]["qas"]:
            event = bm25.search(qa["question"])
            qa["retrieved_event"] = event
        for qa in item["fact"]["local_qas"]:
            event = bm25.search(qa["question"])
            qa["retrieved_event"] = event
    file_name = data_path.split("/")[-1].split(".")[0] + "-with-retrieval.json"
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(data, f, indent=4)


def search_for_tendency(data_path, bm25, save_dir):
    data = json.load(open(data_path))
    for item in tqdm(data):
        for qa in item["tendency"]["qas"]:
            event = bm25.search(qa["question"])
            qa["retrieved_event"] = event
        for qa in item["tendency"]["local_qas"]:
            event = bm25.search(qa["question"])
            qa["retrieved_event"] = event
    file_name = data_path.split("/")[-1].split(".")[0] + "-with-retrieval.json"
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    # data_path = "../data/original/test.json"
    # save_dir = Path("data/bm25/fact-test")
    # save_dir.mkdir(exist_ok=True, parents=True)

    # event_corpus = load_fact_corpus(data_path)
    # bm25 = BM25Retriever(event_corpus)
    # search_for_fact(data_path, bm25, save_dir)

    data_path = "../data/original/test.json"
    save_dir = Path("data/bm25/tendency")
    save_dir.mkdir(exist_ok=True, parents=True)

    event_corpus = load_tendency_corpus(data_path)
    bm25 = BM25Retriever(event_corpus)
    search_for_tendency(data_path, bm25, save_dir)