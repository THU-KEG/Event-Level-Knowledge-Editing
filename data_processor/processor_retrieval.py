
import os
import json
import random
from pathlib import Path


class FactDataProcessor():
    def __init__(self, test_file, save_dir, save_name, sample=1.0, only_local=False) -> None:
        self.data = json.load(open(test_file))
        if int(sample) != 1:
            print(f"Sample {sample*100}% data from original file.")
            self.data = random.sample(self.data, k=int(sample*len(self.data)))
        self.only_local = only_local
        
        # save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        if only_local:
            self.save_path = os.path.join(save_dir, f"{save_name}_local.json")
        else:
            self.save_path = os.path.join(save_dir, f"{save_name}.json")
    
        # save data
        if only_local:
            instruction = "Please answer the question based on your knowledge. "
            instruction += "Please only output a noun (usually an entity) as the answer, and do not output a complete sentence."
        else:
            instruction = "Given an event, assuming that the event has occurred, please answer the corresponding questions based on the event and your knowledge. "
            instruction += "If you do not know the answer to the question, please respond with 'unknown'. "
            instruction += "Please only output a noun (usually an entity) as the answer, and do not output a complete sentence."
    
        self.final_data = {
            "prompt": {
                "instructions": instruction,
                "input_prefix": "",
                "input_suffix": "\n",
                "output_prefix": "",
                "output_suffix": "\n",
                "demonstrations": []
            },
            "request_states": [
            ]
        }
    

    def get_input(self, item, qa):
        if self.only_local:
            input = f"Question: {qa['question']}\nAnswer:"
        else:
            input = f"Event: {qa['retrieved_event']}\nQuestion: {qa['question']}\nAnswer:"
        return input
    

    def process(self):
        for item in self.data:
            for qa in item["fact"]["local_qas"]:
                input = self.get_input(item, qa)
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
                        "retrieved_event": qa["retrieved_event"],
                        "input": {
                            "text": input
                        },
                        "id": item["id"],
                        "answer": qa["answer"],
                        "local": True,
                        "question_type": "NA"
                    },
                    "request": {
                        "result": {
                            "success": False,
                            "completions": [
                                {
                                    "text": ""
                                }
                            ]
                        },
                        "request_time": 0,
                        "request_datetime": 0
                    }
                }
                self.final_data["request_states"].append(instance)
            
            # only local qas
            if self.only_local:
                continue
            for qa in item["fact"]["qas"]:
                input = self.get_input(item, qa)
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
                        "retrieved_event": qa["retrieved_event"],
                        "input": {
                            "text": input
                        },
                        "id": item["id"],
                        "answer": qa["answer"],
                        "local": False,
                        "question_type": qa["type"]
                    },
                    "request": {
                        "result": {
                            "success": False,
                            "completions": [
                                {
                                    "text": ""
                                }
                            ]
                        },
                        "request_time": 0,
                        "request_datetime": 0
                    }
                }
                self.final_data["request_states"].append(instance)

    def save_data(self):
        with open(self.save_path, "w") as f:
            json.dump(self.final_data, f, indent=4)



class TendencyDataProcessor():
    def __init__(self, test_file, save_dir, save_name, sample=1.0, multi_choice=True) -> None:
        self.data = json.load(open(test_file))
        if int(sample) != 1:
            print(f"Sample {sample*100}% data from original file.")
            self.data = random.sample(self.data, k=int(sample*len(self.data)))
        self.multi_choice = multi_choice
        
        # save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        if multi_choice:
            self.save_path = os.path.join(save_dir, f"{save_name}.json")
        else:
            self.save_path = os.path.join(save_dir, f"{save_name}.json")
        
        # save data
        if self.multi_choice:
            instruction = "Given an event, assuming that the event has occurred, please answer the corresponding questions based on the event and your knowledge. "
            # instruction += "Please do not over-reason the impact of the event. If the event may not have an appreciable impact on the question, please answer C. "
            instruction += "Please only output the option A, B, or C as the answer, and do not output brackets. Do not output a complete sentence or the full answer span."
        else:
            instruction = "Given an event, assuming that the event has occurred, please answer the corresponding questions based on the event and your knowledge. "
            # instruction += "Please do not over-reason the impact of the event. If the event does not have a significant effect on the question, please answer no significant impact and provide some reasons."

        
        self.final_data = {
            "prompt": {
                "instructions": instruction,
                "input_prefix": "",
                "input_suffix": "\n",
                "output_prefix": "",
                "output_suffix": "\n",
                "demonstrations": []
            },
            "request_states": [
            ]
        }
    

    def get_input(self, item, qa):
        if self.multi_choice:
            input = f"Event: {qa['retrieved_event']}\nQuestion: {qa['question']}\n{qa['candidate']}\nAnswer:"
        else:
            input = f"Event: {qa['retrieved_event']}\nQuestion: {qa['question']}\nAnswer:"
        return input
    

    def parse_candidate(self, candidate):
        import re
        spans = re.split("\([ABC]\)", candidate)
        spans = [span.strip() for span in spans if span.strip() != ""]
        assert len(spans) == 3
        options = {
            "A": spans[0],
            "B": spans[1],
            "C": spans[2]
        }
        return options


    def process(self):
        for item in self.data:
            for qa in item["tendency"]["local_qas"]:
                input = self.get_input(item, qa)
                options = self.parse_candidate(qa["candidate"])
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
                        "retrieved_event": qa["retrieved_event"],
                        "input": {
                            "text": input
                        },
                        "id": item["id"],
                        "answer": qa["answer"] if self.multi_choice else options[qa["answer"]],
                        "local": True,
                    },
                    "request": {
                        "result": {
                            "success": False,
                            "completions": [
                                {
                                    "text": ""
                                }
                            ]
                        },
                        "request_time": 0,
                        "request_datetime": 0
                    }
                }
                self.final_data["request_states"].append(instance)
            
            for qa in item["tendency"]["qas"]:
                input = self.get_input(item, qa)
                options = self.parse_candidate(qa["candidate"])
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
                        "retrieved_event": qa["retrieved_event"],
                        "input": {
                            "text": input
                        },
                        "id": item["id"],
                        "answer": qa["answer"] if self.multi_choice else options[qa["answer"]],
                        "local": False
                    },
                    "request": {
                        "result": {
                            "success": False,
                            "completions": [
                                {
                                    "text": ""
                                }
                            ]
                        },
                        "request_time": 0,
                        "request_datetime": 0
                    }
                }
                self.final_data["request_states"].append(instance)

    def save_data(self):
        with open(self.save_path, "w") as f:
            json.dump(self.final_data, f, indent=4)



if __name__ == "__main__":
    # processor = FactDataProcessor(test_file="../retrieval/data/multilingual-e5-large/fact/test-with-retrieval.json", 
    #                           save_dir="../data/processed/fact/gemini-pro", 
    #                           sample=1.0,
    #                           save_name="fact_e5",
    #                           only_local=False)
    # processor.process()
    # processor.save_data()

    models = ["mistral-7b", "gemini-pro", "gpt-3.5", "gpt-4"]
    # models = ["gemini-pro"]
    for model in models:
        processor = TendencyDataProcessor(test_file="../retrieval/data/multilingual-e5-large/tendency/test-with-retrieval.json",
                                save_dir=f"../data/processed/tendency/{model}/local", 
                                sample=1.0, 
                                save_name="tendency_mc_e5",
                                multi_choice=True)
        processor.process()
        processor.save_data()