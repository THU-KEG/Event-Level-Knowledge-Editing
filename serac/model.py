import torch
import transformers
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = BertClassifier(model_args.model_name_or_path)
    if model_args.checkpoint_path is not None:
        model.load_state_dict(torch.load(model_args.checkpoint_path))
    return tokenizer, model


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name_or_path, hidden_dim=768):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.config.problem_type = "multi_label_classification"
        # self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, input_ids, attention_mask, labels):
        model_output = self.model(input_ids, attention_mask, labels=labels.unsqueeze(1), return_dict=True)
        # if "pooler_output" in model_output.keys():
        #     score = self.classifier(model_output.pooler_output).squeeze(1)
        # else:
        #     score = self.classifier(model_output.last_hidden_state[:, 0]).squeeze(1)
        
        # loss_fn = torch.nn.BCEWithLogitsLoss()
        # loss = loss_fn(score, labels)
        logits = torch.sigmoid(model_output["logits"].squeeze(1))
        # import pdb; pdb.set_trace()
        return dict(loss=model_output["loss"], logits=logits)