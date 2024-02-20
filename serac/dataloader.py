import json
from torch.utils import data
from tqdm import tqdm, trange
import random
import torch


class ClassifierDataset(data.Dataset):
    def __init__(self,
                 config,
                 tokenizer: str,
                 input_file: str) -> None:
        """Constructs an EDTCProcessor."""
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.edits = []
        self.questions = []
        self.input_features = []
        self.maximum = 500000
        self.read_examples(input_file)
        self.convert_examples_to_features()
    

    def __len__(self):
        return self.maximum


    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EDInputExample`s for the dataset."""
        with open(input_file, "r") as f:
            data = json.load(f)
            for item in data:
                self.edits.append(item["event"])
                for type in ["fact", "tendency"]:
                    if type not in item:
                        continue
                    for qa in item[type]["qas"]:
                        self.questions.append({
                            "event": item["event"],
                            "question": qa["question"]
                        })
                    for qa in item[type]["local_qas"]:
                        self.questions.append({
                            "event": "NA",
                            "question": qa["question"]
                        })

    def encode(self, edit, question):
        text = f"Event: {edit} Question: {question}"
        outputs = self.tokenizer(text, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.config.max_seq_length,
                        add_special_tokens=True)
        return outputs


    def convert_examples_to_features(self) -> None:
        num_neg_edits = 15
        for i in trange(self.maximum):
            question = random.sample(self.questions, k=1)[0]
            neg_edits = random.sample(self.edits, k=num_neg_edits)
            instance = [] # (input_ids, attention_mask, label)
            gold_event = question["event"]
            if gold_event == "NA":
                neg_event = random.sample(self.edits, k=1)[0]
                outputs = self.encode(neg_event, question["question"])
                instance.append([outputs["input_ids"], outputs["attention_mask"], 0])
            else:
                outputs = self.encode(gold_event, question["question"])
                instance.append([outputs["input_ids"], outputs["attention_mask"], 1])

            for neg_edit in neg_edits:
                label = 0
                if neg_edit == gold_event:
                    label = 1
                outputs = self.encode(neg_edit, question["question"])
                instance.append([outputs["input_ids"], outputs["attention_mask"], label])
            self.input_features.append(instance)
    

    def __getitem__(self,
                    index: int):
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.stack([torch.tensor(item[0], dtype=torch.long) for item in features]),
            attention_mask=torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in features]),
            labels=torch.stack([torch.tensor(item[2], dtype=torch.float32) for item in features])
        )
        return data_dict



class InferClassifierDataset(data.Dataset):
    def __init__(self,
                 config,
                 tokenizer: str,
                 input_file: str) -> None:
        """Constructs an EDTCProcessor."""
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.edits = []
        self.questions = []
        self.input_features = []
        self.read_examples(input_file)
        self.convert_examples_to_features()
    

    def __len__(self):
        return len(self.input_features)


    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EDInputExample`s for the dataset."""
        with open(input_file, "r") as f:
            data = json.load(f)
            for item in data:
                self.edits.append(item["event"])
                for type in ["fact", "tendency"]:
                    if type not in item:
                        continue
                    for qa in item[type]["qas"]:
                        self.questions.append({
                            "event": item["event"],
                            "question": qa["question"]
                        })
                    for qa in item[type]["local_qas"]:
                        self.questions.append({
                            "event": "NA",
                            "question": qa["question"]
                        })

    def encode(self, edit, question):
        text = f"Event: {edit} Question: {question}"
        outputs = self.tokenizer(text, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.config.max_seq_length,
                        add_special_tokens=True)
        return outputs


    def convert_examples_to_features(self) -> None:
        for question in tqdm(self.questions):
            for event in self.edits:
                outputs = self.encode(event, question["question"])
                label = 0
                if event == question["event"]: label = 1
                self.input_features.append([outputs["input_ids"], outputs["attention_mask"], label])
        print(len(self.input_features))
    

    def __getitem__(self,
                    index: int):
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.tensor(features[0], dtype=torch.long).unsqueeze(0),
            attention_mask=torch.tensor(features[1], dtype=torch.float32).unsqueeze(0),
            labels=torch.tensor(features[2], dtype=torch.float32).unsqueeze(0)
        )
        return data_dict



def collate_fn(batch):
    """Collates the samples in batches."""
    output_batch = dict()
    for key in batch[0].keys():
        output_batch[key] = torch.cat([x[key] for x in batch], dim=0)
    return output_batch

            
