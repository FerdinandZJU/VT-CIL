EPOCHS=$1
FE_LEARNING_RATE=$2
DATASET=$3
SEED=$4
CUDA_DEVICES=$5

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/joint/${DATASET}


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python joint.py \
                        --epochs $EPOCHS \
                        --dataset $DATASET \
                        --seed $SEED \
                        --fe_learning_rate $FE_LEARNING_RATE \
                        2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/joint/${DATASET}/log.log