MODEL="gpt-3.5" # specify the model

cd data_processor

# process for IKE
python processor.py --model $MODEL

# process for ft
python processor_ft.py

# process for serac
python processor_serac.py --model $MODEL

# process for retrieval
python processor_retrieval.py --model $MODEL

