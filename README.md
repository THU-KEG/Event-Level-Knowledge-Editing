Dataset and code for "Event-Level Knowledge Editing". Event-level knowledge editing aims at editing newly occurred events into LLMs, thereby updating multiple factual knowledge and influenced tendencies at once. We construct a high-quality event-level editing benchmark $ELKEN$, consisting of `1,515` event edits, `6,449` questions about factual knowledge, and `10,150` questions about future tendencies. 

![Event-Level Knowledge Editing](./imgs/figure1.png)

## 1. Quick Start
The code repository is based on Pytorch and Transformers. Please use the following command to install the necessary dependcies. `pip install -r requirements.txt`.

## 2. Data Processor
Using `./data_processor/processor.py` to process data for the method ICE (In-context Editing)

For the retrieval method, please first retrieve events for each question using codes in `./retrieval`. And then using `./data_processor/processor_retrieval.py` to process data.

For the SERAC method, please first use the codes in `./serac` to train a scope classifier and determine whether a question needs an retrieved event. And then using `./data_processor/processor_serac.py` to process data.

For the fine-tuning method, please first use the codes in `./fine-tuning` to fine-tune an LLM using all the events in test set with a language modelling object. And then using `./data_processor/processor_ft.py` to process data.

## 3. Run Experiments
Using the codes in the folder `./llm` to evaluate GPT-3.5, GPT-4, and Gemini Pro. You need to place your api key in corresponding files.
Using the codes in the folder `./open-source` to evaluate GPT-J, TULU 2, and Mistral 7B.

## 4. Evaluation
Using `metric.py ` to get the reliability and locality for question-level and event-level metrics for factual knowledge and tendency separately.

Using `overall_metric.py` to get the overall event-level metrics, i.e., an event edit is reliable if and only if all the corresponding questions about factual knowledge and tendencies are correct.

