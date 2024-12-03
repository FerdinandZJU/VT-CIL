DATASET=$1
EPOCH=$2
LEARNING_RATE=$3
CUDA_DEVICES=$4

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/${DATASET}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python pretrain_resnet.py \
                                          --dataset $DATASET \
                                          --epochs $EPOCH \
                                          --learning_rate $LEARNING_RATE \
                                          2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/${DATASET}/log.log