"""
This script contains an example how to perform semantic search with PyTorch. It performs exact nearest neighborh search.

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions (we only use about 100k):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs


As embeddings model, we use the SBERT model 'quora-distilbert-multilingual',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).


Google Colab example: https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time

import json
from pathlib import Path
from tqdm import tqdm
from utils import load_fact_corpus, load_tendency_corpus



class DenseRetriever():
    def __init__(self, model_name, corpus_sentences, data_type, save_dir) -> None:
        self.model = SentenceTransformer(model_name).to("cuda:6")
        self.embedding_cache_path = os.path.join(save_dir, "{}-embeddings-{}.pkl".format(data_type, model_name.split("/")[-1]))
        self.corpus_embeddings = self.index(corpus_sentences).to(self.model.device)
        self.corpus_sentences = corpus_sentences


    def index(self, corpus_sentences):
        # Check if embedding cache path exists
        if not os.path.exists(self.embedding_cache_path):
            print("Encode the corpus. This might take a while")
            corpus_embeddings = self.model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

            print("Store file on disc")
            with open(self.embedding_cache_path, "wb") as fOut:
                pickle.dump({"sentences": corpus_sentences, "embeddings": corpus_embeddings}, fOut)
        else:
            print("Load pre-computed embeddings from disc")
            with open(self.embedding_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_sentences = cache_data["sentences"]
            corpus_embeddings = cache_data["embeddings"]
        ###############################
        print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))
        return corpus_embeddings


    def search(self, query, n=1):
        start_time = time.time()
        question_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, self.corpus_embeddings)
        end_time = time.time()
        hits = hits[0]  # Get the hits for the first query

        # print("Input question:", query)
        # print("Results (after {:.3f} seconds):".format(end_time - start_time))
        # for hit in hits[0:n]:
        #     print("\t{:.3f}\t{}".format(hit["score"], self.corpus_sentences[hit["corpus_id"]]))

        return self.corpus_sentences[hits[0]["corpus_id"]], hits[0]["score"]


def search_for_fact(data_path, retriever, save_dir):
    data = json.load(open(data_path))
    for item in tqdm(data):
        for qa in item["fact"]["qas"]:
            event = retriever.search(qa["question"])
            qa["retrieved_event"] = event
        for qa in item["fact"]["local_qas"]:
            event, score = retriever.search(qa["question"])
            qa["retrieved_event"] = event
    file_name = data_path.split("/")[-1].split(".")[0] + "-with-retrieval.json"
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(data, f, indent=4)


def search_for_tendency(data_path, retriever, save_dir):
    data = json.load(open(data_path))
    for item in tqdm(data):
        for qa in item["tendency"]["qas"]:
            event, score = retriever.search(qa["question"])
            qa["retrieved_event"] = event
            qa["retrieved_score"] = score
        for qa in item["tendency"]["local_qas"]:
            event, score = retriever.search(qa["question"])
            qa["retrieved_event"] = event
            qa["retrieved_score"] = score
    file_name = data_path.split("/")[-1].split(".")[0] + "-with-retrieval.json"
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    # model_name = "/data3/MODELS/multilingual-e5-large"
    # data_path = "../data/test.json"
    # save_dir = Path("data/{}".format(model_name.split("/")[-1]))
    # save_dir.mkdir(exist_ok=True, parents=True)

    # corpus_sentences = load_fact_corpus(data_path)
    # data_type = "test-fact"
    # retriever = DenseRetriever(model_name, corpus_sentences, data_type, save_dir)
    # search_for_fact(data_path, retriever, save_dir)

    model_name = "/data3/MODELS/multilingual-e5-large"
    data_path = "../data/original/test.json"
    save_dir = Path("data/{}/tendency".format(model_name.split("/")[-1]))
    save_dir.mkdir(exist_ok=True, parents=True)

    corpus_sentences = load_tendency_corpus(data_path)
    data_type = "test"
    retriever = DenseRetriever(model_name, corpus_sentences, data_type, save_dir)
    search_for_tendency(data_path, retriever, save_dir)