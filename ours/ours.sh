DATASET=$1
if [ "$DATASET" == "TaG" ]; then
    EPOCHS=60
    FE_LEARNING_RATE=0.001
    EXEMPLARS=300
    L1=0.07
    L2=1.0
    L3=3.0
    SEED=42
    CUDA_DEVICES=0
elif [ "$DATASET" == "OFR" ]; then
    EPOCHS=30
    FE_LEARNING_RATE=0.01
    EXEMPLARS=60
    L1=0.005
    L2=1.0
    L3=0.8
    SEED=20
    CUDA_DEVICES=0
else
    echo "Error: Unknown dataset '$DATASET'. Please use 'TaG' or 'OFR'."
    exit 1
fi

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/ours/${DATASET}
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ours.py \
                        --epochs $EPOCHS \
                        --dataset $DATASET \
                        --seed $SEED \
                        --lambda_1 $L1 \
                        --lambda_2 $L2 \
                        --lambda_3 $L3 \
                        --memory_size $EXEMPLARS \
                        --fe_learning_rate $FE_LEARNING_RATE \
                        2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/ours/${DATASET}/log.log