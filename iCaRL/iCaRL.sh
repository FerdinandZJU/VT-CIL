EPOCHS=$1
FE_LEARNING_RATE=$2
DATASET=$3
SEED=$4
EXEMPLAR=$5
CUDA_DEVICES=$6


mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/iCaRL/${DATASET}/


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python iCaRL.py \
                        --epochs $EPOCHS \
                        --dataset $DATASET \
                        --seed $SEED \
                        --memory_size $EXEMPLAR \
                        --fe_learning_rate $FE_LEARNING_RATE \
                        2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/iCaRL/${DATASET}/log.log