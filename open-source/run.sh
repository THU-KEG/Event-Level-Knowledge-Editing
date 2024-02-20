# MODEL=$1
# PORT=$2
# HOST=$3

# deepspeed --master_port=$PORT --include localhost:$HOST \
#     main.py config/$1.yaml

MODEL=$2
CUDA_VISIBLE_DEVICES=$1 python temp.py \
    --model_path $2