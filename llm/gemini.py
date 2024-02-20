import json
import time
from tqdm import tqdm
import argparse
from multiprocessing import Pool, Lock

import google.generativeai as genai


GOOGLE_API_KEY=""
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')


def query_openai_api_per_example(args,
                                 prompt, 
                                 instance,
                                 sleep_second,
                                 max_tokens,
                                 demonstrations=None):
    input = instance["instance"]["input"]["text"]
    s_time = time.time()
    success = False
    if prompt["instructions"] != "": ## fact or tendency
        input = prompt["instructions"] + " " + input
    # else:
    #     input = input
    # _, input = input.split("\n\n")
    while not success:
        try:
            # chat_completion = model.generate_content(messages[-1]["content"])
            chat_completion = model.generate_content(input)
        except Exception as e:
            if args.debug:
                import pdb; pdb.set_trace()
            print(e)
            time.sleep(sleep_second)
        else:
            success = True
    try:
        result = chat_completion.text
    except:
        print("\n\n ------ \n\n")
        print("An error raise")
        if args.debug:
            import pdb; pdb.set_trace()
        # print(chat_completion.candidates, chat_completion.parts)
        result = "NA"


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
    args.n_threads = min(1, args.n_threads)
    # demonstrations
    examples = []
    
    with open(args.test_file, 'r+') as f:
        data = json.load(f)
        prompt = data['prompt']
        examples = prompt["demonstrations"]
        batch = []
        if args.debug:
            for item in tqdm(data['request_states']):
                if not item['request']['result'].get('success', False):
                    query_openai_api_per_example(args, prompt, item, args.sleep_second, args.max_tokens, examples)
        else:
            for sample in tqdm(data['request_states']):
                if not sample['request']['result'].get('success', False):
                    batch.append(sample)
                    if len(batch) == args.n_threads:
                        with Pool(args.n_threads) as p:
                            requests = p.starmap(query_openai_api_per_example, [(args, prompt, batch[i], args.sleep_second, args.max_tokens, examples) for i in range(args.n_threads)])
                        for b, request in zip(batch, requests):
                            b['request'] = request
                        batch.clear()
                        f.seek(0)
                        json.dump(data, f, indent=4)
                        f.truncate()
            
            # process the rest samples
            if len(batch) != 0:
                with Pool(args.n_threads) as p:
                    requests = p.starmap(query_openai_api_per_example, [(args, prompt, batch[i], args.sleep_second, args.max_tokens, examples) for i in range(len(batch))])
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
    # parser.add_argument("--model", type=str, default="gpt-4-1106")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sleep_second", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
    