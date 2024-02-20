# CUDA_VISIBLE_DEVICES=$1 python run-clm.py \
#     --model_name_or_path /data3/MODELS/Mistral-7B-Instruct-v0.2 \
#     --train_file events.txt \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --do_train \
#     --learning_rate 3e-5 \
#     --output_dir output/mistral-7b \
#     --dataloader_num_workers 1 \
#     --dataloader_prefetch_factor 1 \
#     --overwrite_output_dir \
#     --bf16
    # --push_to_hub

deepspeed --master_port=9904 --include localhost:1,2,3,7 run-clm.py \
    --model_name_or_path /data3/MODELS/tulu-v2-7b \
    --train_file events.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --learning_rate 3e-5 \
    --output_dir output/tulu-v2-7b \
    --dataloader_num_workers 1 \
    --dataloader_prefetch_factor 1 \
    --overwrite_output_dir \
    --bf16 \
    --deepspeed zero_2.json