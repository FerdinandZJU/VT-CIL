DATASET=$1
if [ "$DATASET" == "TaG" ]; then
    EPOCH=240
    LEARNING_RATE=0.03
    CUDA_DEVICES=0
elif [ "$DATASET" == "OFR" ]; then
    EPOCH=240
    LEARNING_RATE=0.001
    CUDA_DEVICES=0
else
    echo "Error: Unknown dataset '$DATASET'. Please use 'TaG' or 'OFR'."
    exit 1
fi

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/${DATASET}
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python pretrain_resnet.py \
                                          --dataset $DATASET \
                                          --epochs $EPOCH \
                                          --learning_rate $LEARNING_RATE \
                                          2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/${DATASET}/log.log