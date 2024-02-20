import os
import re
import json
from collections import defaultdict, Counter

def parse_answer(text):
    text = text.split("\n")[0].strip().lower()
    if text != "" and text[-1] == ".":
        return text[:-1]
    return text

def parse_local_answer(text, alias=None):
    text = parse_answer(text)
    if alias is not None:
        if text in alias:
            text = alias[0]
    if "unknown" in text:
        return "unknown"
    else:
        return text

def parse_tendency_answer(text):
    # text = text.strip()
    text = text.split("\n")[0].strip()
    if "(" in text:
        pattern = re.compile("\(([ABC])\)")
        spans = pattern.findall(text)
        if len(spans) != 1:
            return text
        else:
            return spans[0]
    elif "." in text:
        return text.split(".")[0]
    else:
        return text


def get_question(item):
    if "\n\n" not in item["instance"]["input"]["text"]:
        if len(item["instance"]["input"]["text"].split("\n")) == 3:
            event, question, _ = item["instance"]["input"]["text"].split("\n")
        else: # ft
            event = "NA"
            question, _  = item["instance"]["input"]["text"].split("\n")
    else:
        input = item["instance"]["input"]["text"].split("\n\n")[-1]
        if len(input.split("\n")) == 3:
            event, question, _ = input.split("\n")
        else: # serac
            event = "NA"
            question, _ = input.split("\n")
    return question.replace("Question: ", "")

class MetricForFact():
    def __init__(self, predictions, labels, data, locality_data=None) -> None:
        self.predictions = predictions
        self.labels = labels
        self.data = data
        self.locality_data = locality_data
        self.metrics = {}
        self.properties = self.get_properties()

    @staticmethod
    def load_file(answer_file):
        predictions, labels = [], []
        data = json.load(open(answer_file))
        for item in data["request_states"]:
            predictions.append(item["request"]["result"]["completions"][0]["text"])
            labels.append(item["instance"]["answer"]["alias"] + [item["instance"]["answer"]["name"]])
            if item["instance"]["answer"]["id"] == "Q30":
                labels[-1].append("United States")
        return data, predictions, labels


    def get_properties(self):
        properties = defaultdict(list)
        for item in self.data["request_states"]:
            properties["question_types"].append(item["instance"]["question_type"])
            properties["event_types"].append(item["instance"]["event_type"])
            properties["local"].append(item["instance"]["local"])
            properties["event"].append(item["instance"]["event"])
        return properties


    def compute_acc(self):
        counter = Counter()
        # overall
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if self.properties["local"][i]: continue
            question_type = self.properties["question_types"][i]
            correct = 0
            # unknown
            if "unknown" in label:
                counter["unknown_total"] += 1
                if "unknown" in parse_answer(pred):
                    counter["unknown_correct"] += 1
                    correct = 1
            else:
                counter["known_total"] += 1
                if parse_answer(pred) in [parse_answer(_label) for _label in label]:
                    counter["known_correct"] += 1
                    correct = 1
            # question type
            if question_type == 0:
                counter["0_total"] += 1
                counter["0_correct"] += correct
            elif question_type == 1:
                if "unknown" not in label:
                    counter["1_total"] += 1
                    counter["1_correct"] += correct
            else:
                raise ValueError

            # overall
            counter["overall_total"] += 1
            counter["overall_correct"] += correct
        
        for type in ["unknown", "known", "0", "1", "overall"]:
            counter[f"{type}_acc"] = counter[f"{type}_correct"]/ counter[f"{type}_total"]
        self.metrics["acc"] = counter
        print(counter)
    

    def compute_event_acc(self):
        counter = Counter()
        flag_event = "NA"
        correct = 1
        # overall
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if self.properties["local"][i]: continue
            if self.properties["event"][i] != flag_event:
                if flag_event != "NA":
                    counter["overall_total"] += 1
                    counter["overall_correct"] += correct
                else:
                    pass
                flag_event = self.properties["event"][i]
                correct = 1
            if "unknown" in label:
                if "unknown" in parse_answer(pred):
                    correct = 1 * correct
                else:
                    correct = 0
            else:
                if parse_answer(pred) in [parse_answer(_label) for _label in label]:
                    correct = 1 * correct
                else:
                    correct = 0
        counter["overall_total"] += 1
        counter["overall_correct"] += correct
        counter["overall_acc"] = counter["overall_correct"] / counter["overall_total"]
        self.metrics["event_acc"] = counter
        print(counter)

    
    def get_input_pred(self, data):
        input2pred = {}
        all_alias = {}
        for item in data["request_states"]:
            if not item["instance"]["local"]: continue
            if "\n\n" in item["instance"]["input"]["text"]: # SERAC
                item["instance"]["input"]["text"] = item["instance"]["input"]["text"].split("\n\n")[-1]
            if "Event:" in item["instance"]["input"]["text"]: # SERAC
                question = "\n".join(item["instance"]["input"]["text"].split("\n")[1:])
            else:
                question = item["instance"]["input"]["text"]
            assert question not in input2pred
            input2pred[question] = item["request"]["result"]["completions"][0]["text"]
            all_alias[question] = item["instance"]["answer"]["alias"]
        return input2pred, all_alias


    def compute_locality(self):
        input2pred, all_alias = self.get_input_pred(self.data)
        counter = Counter()
        assert len(input2pred) == len(self.locality_data["request_states"])
        for item in self.locality_data["request_states"]:
            assert item["instance"]["input"]["text"] in input2pred
            assert item["instance"]["local"]
            alias = all_alias[item["instance"]["input"]["text"]]
            answer_after_editing = input2pred[item["instance"]["input"]["text"]]
            answer_before_editing = item["request"]["result"]["completions"][0]["text"]
            counter["overall_total"] += 1
            if parse_local_answer(answer_after_editing, alias) == parse_local_answer(answer_before_editing, alias):
                counter["overall_same"] += 1
        counter["overall_consistency"] = counter["overall_same"] / counter["overall_total"]
        print(counter)
        self.metrics["locality"] = counter


    def dump_metrics(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)



class MetricForTendency():
    def __init__(self, predictions, labels, data, locality_data=None) -> None:
        self.predictions = predictions
        self.labels = labels
        self.data = data
        self.locality_data = locality_data
        self.metrics = {}
        self.properties = self.get_properties()


    @staticmethod
    def load_file(answer_file):
        predictions, labels = [], []
        data = json.load(open(answer_file))
        for item in data["request_states"]:
            predictions.append(item["request"]["result"]["completions"][0]["text"])
            labels.append(item["instance"]["answer"])
        return data, predictions, labels


    def get_properties(self):
        properties = defaultdict(list)
        for item in self.data["request_states"]:
            properties["event_types"].append(item["instance"]["event_type"])
            properties["local"].append(item["instance"]["local"])
            properties["question_type"].append(item["instance"].get("question_type", "NA"))
            properties["event"].append(item["instance"]["event"])
        return properties


    def compute_acc(self):
        counter = Counter()
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if self.properties["local"][i]:
                continue
            counter["overall_total"] += 1
            if parse_tendency_answer(pred) == parse_tendency_answer(label):
                counter["overall_correct"] += 1
            else:
                print(pred, label)
        counter["overall_acc"] = counter["overall_correct"] / (counter["overall_total"] + 1e-10)
        self.metrics["acc"] = counter

        print(self.metrics)
    

    def compute_event_acc(self):
        counter = Counter()
        flag_event = "NA"
        correct = 1
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if self.properties["local"][i]: continue
            if self.properties["event"][i] != flag_event:
                if flag_event != "NA":
                    counter["overall_total"] += 1
                    counter["overall_correct"] += correct
                else:
                    pass
                flag_event = self.properties["event"][i]
                correct = 1
            if parse_tendency_answer(pred) == parse_tendency_answer(label):
                correct = 1 * correct
            else:
                correct = 0
        counter["overall_total"] += 1
        counter["overall_correct"] += correct
        counter["overall_acc"] = counter["overall_correct"] / counter["overall_total"]
        self.metrics["event_acc"] = counter
        print(counter)
    


    def get_input_pred(self, data):
        input2pred = []
        for item in data["request_states"]:
            if not item["instance"]["local"]: continue
            if "\n\n" in item["instance"]["input"]["text"]: # SERAC
                item["instance"]["input"]["text"] = item["instance"]["input"]["text"].split("\n\n")[-1]
            if "Event:" in item["instance"]["input"]["text"]: # SERAC
                question = "\n".join(item["instance"]["input"]["text"].split("\n")[1:])
            else:
                question = item["instance"]["input"]["text"]
            input2pred.append([question, item["request"]["result"]["completions"][0]["text"]])
        return input2pred


    def compute_locality(self):
        input2pred = self.get_input_pred(self.data)
        counter = Counter()
        assert len(input2pred) == len(self.locality_data["request_states"])
        for i, item in enumerate(self.locality_data["request_states"]):
            assert item["instance"]["local"]
            # import pdb; pdb.set_trace()
            assert item["instance"]["input"]["text"].split("\n")[0] == input2pred[i][0].split("\n")[0]
            answer_after_editing = input2pred[i][1]
            answer_before_editing = item["request"]["result"]["completions"][0]["text"]
            counter["overall_total"] += 1
            if parse_local_answer(answer_after_editing) == parse_local_answer(answer_before_editing):
                counter["overall_same"] += 1
        counter["overall_consistency"] = counter["overall_same"] / counter["overall_total"]
        print(counter)
        self.metrics["locality"] = counter


    def dump_metrics(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
    


class MetricForAutoEval():
    def __init__(self, data) -> None:
        self.data = data
        self.metrics = {}


    @staticmethod
    def load_file(answer_file):
        data = json.load(open(answer_file))
        return data


    def compute_score(self):
        # not local; reliablity
        counter = Counter() # not local
        total = 0
        for item in self.data["request_states"]:
            if item["instance"]["local"]:
                continue
            total += 1
            if item["scores"]["overall"] == 5:
                counter["full_mark"] += 1
            if item["scores"]["accuracy"] == 5:
                counter["acc_full_mark"] += 1
            if item["scores"]["coherence"] == 5:
                counter["cohere_full_mark"] += 1
            if item["scores"]["comprehensive"] == 5:
                counter["compre_full_mark"] += 1
            
            for key in item["scores"]:
                counter[key] += item["scores"][key]

        for key in counter:
            counter[key] /= total
        counter["total"] = total
        self.metrics["score"] = counter

        print(self.metrics)
    

    def compute_event_score(self):
        # not local; reliablity
        counter = Counter() # not local
        flag_event = "NA"
        flag = 1
        counter_per_event = Counter()
        for item in self.data["request_states"]:
            if item["instance"]["local"]: continue
            if item["instance"]["event"] != flag_event:
                if flag_event != "NA":
                    for key in counter_per_event:
                        counter_per_event[key] /= counter_per_event["total"]
                    assert int(counter_per_event["total"]) == 1 or len(counter_per_event) == 0
                    for key in counter_per_event:
                        counter[key] += counter_per_event[key]
                    counter["full_mark"] += flag
                else:
                    pass
                flag_event = item["instance"]["event"]
                counter_per_event = Counter()
                flag = 1
            if item["scores"]["overall"] != 5:
                flag = 0
            if item["scores"]["overall"] == -1: continue
            for key in item["scores"]:
                counter_per_event[key] += item["scores"][key]
            counter_per_event["total"] += 1
        
        for key in counter_per_event:
            counter[key] += counter_per_event[key]
        print(counter)
        total = counter["total"]
        for key in counter:
            counter[key] /= total
        self.metrics["event_score"] = counter


    def dump_metrics(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)



if __name__ == "__main__":
    # models = ["gpt-j-6b", "tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    # # models = ["Mistral-7B-Instruct-v0.2"]
    # # # io_dir = "data/processed/fact/gemini-pro"
    # models = ["gpt-4", "gpt-3.5", "gemini-pro"]
    # # models = ["gpt-3.5"]
    # # files = [
    # #     ("fact_predictions.json", "metrics_icl.json"),
    # #     ("fact_bm25_predictions.json", "metrics_bm25.json"),
    # #     ("fact_e5_predictions.json", "metrics_e5.json"),
    # #     ("fact_serac_predictions.json", "metrics_serac.json"),
    # #     ("fact_ft_predictions.json", "metrics_ft.json") 
    # # ]
    # files = [
    #     ("fact.json", "metrics_icl.json"),
    #     ("fact_bm25.json", "metrics_bm25.json"),
    #     ("fact_e5.json", "metrics_e5.json"),
    #     ("fact_serac.json", "metrics_serac.json")
    # ]
    # for model in models:
    #     for file in files:
    #         # io_dir = f"open-source/output/{model}"
    #         io_dir = f"data/processed/fact/{model}"
    #         # answer_file = "open-source/output/vicuna-7b-full/predictions.json"
    #         answer_file = os.path.join(io_dir, file[0])
    #         data, predictions, labels = MetricForFact.load_file(answer_file)
    #         local_file = os.path.join(io_dir, "fact_local.json")
    #         locality_data, _, _ = MetricForFact.load_file(local_file)
            
    #         metric = MetricForFact(predictions, labels, data, locality_data)
    #         metric.compute_acc()
    #         metric.compute_event_acc()
    #         metric.compute_locality()
    #         metric.dump_metrics(os.path.join(io_dir, file[1]))


    # # models = ["Mistral-7B-Instruct-v0.2"]
    # # models = ["tulu-v2-7b"]
    # models = ["gpt-j-6b"]
    # files = [
    #     ("tendency_mc_predictions.json", "metrics_mc_icl.json"),
    #     ("tendency_mc_bm25_predictions.json", "metrics_mc_bm25.json"),
    #     ("tendency_mc_e5_predictions.json", "metrics_mc_e5.json"),
    #     ("tendency_mc_serac_predictions.json", "metrics_mc_serac.json"),
    #     ("tendency_mc_ft_predictions.json", "metrics_mc_ft.json")
    # ]
    # for model in models:
    #     for file in files:
    #         io_dir = f"open-source/output/{model}"
    #         answer_file = os.path.join(io_dir, file[0])
    #         data, predictions, labels = MetricForTendency.load_file(answer_file)
    #         local_file = os.path.join(io_dir, "tendency_mc_local_predictions.json")
    #         # local_file = os.path.join(os.path.join(io_dir, "local"), "tendency_mc_local.json")
    #         locality_data, _, _ = MetricForTendency.load_file(local_file)
            
    #         metric = MetricForTendency(predictions, labels, data, locality_data)
    #         metric.compute_acc()
    #         metric.compute_event_acc()
    #         metric.compute_locality()
    #         metric.dump_metrics(os.path.join(io_dir, file[1]))
    
    # models = ["gpt-4", "gpt-3.5", "gemini-pro"]
    models = ["tulu-v2-7b", "Mistral-7B-Instruct-v0.2"]
    models += ["gpt-j-6b"]
    files = [
        ("tendency_gen_exam.json", "metrics_gen_icl.json"),
        ("tendency_gen_bm25_exam.json", "metrics_gen_bm25.json"),
        ("tendency_gen_e5_exam.json", "metrics_gen_e5.json"),
        ("tendency_gen_serac_exam.json", "metrics_gen_serac.json"),
        ("tendency_gen_ft_exam.json", "metrics_gen_ft.json")
    ]
    for model in models:
        for file in files:
            # io_dir = f"data/processed/tendency/{model}/examiner"
            io_dir = f"open-source/output/{model}/examiner"
            answer_file = os.path.join(io_dir, file[0])
            data = MetricForAutoEval.load_file(answer_file)
            metric = MetricForAutoEval(data)
            metric.compute_score()
            metric.compute_event_score()
            metric.dump_metrics(os.path.join(io_dir, file[1]))

