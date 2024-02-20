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
        self.save_path = os.path.join(save_dir, f"fact_recall.json")
    
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
    

    def get_input(self, qa):
        input = f"Question: {qa['question']}\nAnswer:"
        return input
    

    def process(self):
        for qa in self.data:
            input = self.get_input(qa)
            instance =  {
                "instance": {
                    "input": {
                        "text": input
                    },
                    "answer": qa["answer"],
                    "item_id": qa["item_id"],
                    "qa_id": qa["qa_id"]
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
    # models = ["gpt-3.5", "gpt-4", "gemini-pro", "mistral-7b"]
    models = ["gpt-3.5", "gpt-4", "gemini-pro", "mistral-7b"]
    for model in models:
        processor = FactDataProcessor(test_file="qas.json", 
                                save_dir=f"data/{model}",
                                sample=1.0)
        processor.process()
        processor.save_data()