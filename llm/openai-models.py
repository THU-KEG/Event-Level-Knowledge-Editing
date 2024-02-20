import openai
import os
import json
import time
import math
import requests
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Lock

api_key_pool = [ # your api pool
]

url = ""

def query_openai_api_per_example(args,
                                 prompt, 
                                 instance,
                                 model,
                                 sleep_second,
                                 max_tokens,
                                 demonstrations=None):
    # print(api_key)
    input = instance["instance"]["input"]["text"]
    # instructions, input = instance["instance"]["input"]["text"].split("\n\n")
    s_time = time.time()
    success = False
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106"]:
        messages = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant."}
        ]
        messages.append(
            {
                "role": "user",
                "content": prompt["instructions"]
                # "content": instructions
            }
        )
        if demonstrations is not None:
            for example in demonstrations:
                messages.append(
                    {
                        "role": "user",
                        "content": prompt["input_prefix"] + example["input"] + prompt["input_suffix"]
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": prompt["output_prefix"] + example["output"] + prompt["output_suffix"]
                    }
                )
        messages.append({"role": "user", "content": prompt["input_prefix"] + input + prompt["input_suffix"]})
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "user_token": ''
        })
        headers = {'Content-Type': 'application/json'}
        while not success:
            try:
                response = requests.request("POST", url, headers=headers, data=payload, timeout = 15).json()
            except Exception as e:
                if args.debug:
                    import pdb; pdb.set_trace()
                print(e)
                time.sleep(sleep_second)
            else:
                success = True
        result = response['choices'][0]['message']['content']

    instance["request"] = {
        "result": {
            "success": success,
            "completions": [{"text": result}],
        },
        "request_time": time.time() - s_time,
        "request_datetime": time.time()
    }
    return instance['request']


def main(args):
    # demonstrations
    examples = []
    
    with open(args.test_file, 'r+') as f:
        data = json.load(f)
        prompt = data['prompt']
        examples = prompt["demonstrations"]
        batch = []
        if args.debug:
            query_openai_api_per_example(args, prompt, data["request_states"][0], args.model, args.sleep_second, args.max_tokens, examples)
        else:
            for sample in tqdm(data['request_states']):
                if not sample['request']['result'].get('success', False):
                    batch.append(sample)
                    if len(batch) == args.n_threads:
                        with Pool(args.n_threads) as p:
                            requests = p.starmap(query_openai_api_per_example, [(args, prompt, batch[i], args.model, args.sleep_second, args.max_tokens, examples) for i in range(args.n_threads)])
                        for b, request in zip(batch, requests):
                            b['request'] = request
                        batch.clear()
                        f.seek(0)
                        json.dump(data, f, indent=4)
                        f.truncate()
            
            # process the rest samples
            if len(batch) != 0:
                with Pool(args.n_threads) as p:
                    requests = p.starmap(query_openai_api_per_example, [(args, prompt, batch[i], args.model, args.sleep_second, args.max_tokens, examples) for i in range(len(batch))])
                for b, request in zip(batch, requests):
                    b['request'] = request
                batch.clear()
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query OpenAI")
    # multiple processing
    parser.add_argument("--n_threads", type=int, default=4)
    # I/O
    # parser.add_argument("--input_dir", type=str, default="prompts")
    parser.add_argument("--test_file", type=str, default=None)

    # model & parameters
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sleep_second", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
    