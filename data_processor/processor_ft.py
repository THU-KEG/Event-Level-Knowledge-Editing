import os
import json
import random
from pathlib import Path


class FactDataProcessor():
    def __init__(self, test_file, save_dir, sample=1.0) -> None:
        self.data = json.load(open(test_file))
        if int(sample) != 1:
            print(f"Sample {sample*100}% data from original file.")
            self.data = random.sample(self.data, k=int(sample*len(self.data)))
        
        # save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_path = os.path.join(save_dir, f"fact_ft.json")
    
        # save data
        instruction = "Please answer the question based on your knowledge. "
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
        input = f"Question: {qa['question']}\nAnswer:"
        return input
    

    def process(self):
        for item in self.data:
            for qa in item["fact"]["local_qas"]:
                input = self.get_input(item, qa)
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
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
            
            for qa in item["fact"]["qas"]:
                input = self.get_input(item, qa)
                instance =  {
                    "instance": {
                        "event_type": item["event_type"],
                        "event": item["event"],
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
    def __init__(self, test_file, save_dir, sample=1.0, multi_choice=True) -> None:
        self.data = json.load(open(test_file))
        if int(sample) != 1:
            print(f"Sample {sample*100}% data from original file.")
            self.data = random.sample(self.data, k=int(sample*len(self.data)))
        self.multi_choice = multi_choice
        
        # save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        if multi_choice:
            self.save_path = os.path.join(save_dir, f"tendency_mc_ft.json")
        else:
            self.save_path = os.path.join(save_dir, f"tendency_gen_ft.json")
        
        # save data
        if self.multi_choice:
            instruction = "Please answer the question based on your knowledge. "
            instruction += "Please only output the option A, B, or C as the answer, and do not output brackets. Do not output a complete sentence or the full answer span."
        else:
            instruction = "Please answer the question based on your knowledge."

        
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
            input = f"Question: {qa['question']}\n{qa['candidate']}\nAnswer:"
        else:
            input = f"Question: {qa['question']}\nAnswer:"
        return input
    

    def parse_candidate(self, candidate):
        import re
        spans = re.split("\([ABC]\)", candidate)
        spans = [span.strip() for span in spans if span.strip() != ""]
        try:
            assert len(spans) == 3
        except:
            import pdb; pdb.set_trace()
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
    # processor = FactDataProcessor(test_file="../data/original/test.json", 
    #                           save_dir="../data/processed/fact/mistral-7b", 
    #                           sample=1.0)
    # processor.process()
    # processor.save_data()

    processor = TendencyDataProcessor(test_file="../data/original/test.json", 
                              save_dir="../data/processed/tendency/mistral-7b",
                              sample=1.0, 
                              multi_choice=True)
    processor.process()
    processor.save_data()