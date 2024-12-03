EPOCHS=$1
FE_LEARNING_RATE=$2
DATASET=$3
EXEMPLARS=$4
SEED=$5
CUDA_DEVICES=$6

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/SS_IL/${DATASET}


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python SS_IL.py \
                        --epochs $EPOCHS \
                        --dataset $DATASET \
                        --seed $SEED \
                        --memory_size $EXEMPLARS \
                        --fe_learning_rate $FE_LEARNING_RATE \
                        2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/SS_IL/${DATASET}/log.log
