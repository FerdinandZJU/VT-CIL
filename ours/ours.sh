EPOCHS=$1
FE_LEARNING_RATE=$2
DATASET=$3
EXEMPLARS=$4
L1=$5
L2=$6
L3=$7
SEED=$8
CUDA_DEVICES=$9

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