EPOCHS=$1
FE_LEARNING_RATE=$2
DATASET=$3
EXEMPLARS=$4
EXEM_TRAINING=$5
SEED=$6
CUDA_DEVICES=$7

mkdir -p /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/loss_decoupling/${DATASET}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python loss_decoupling.py \
                        --epochs $EPOCHS \
                        --dataset $DATASET \
                        --seed $SEED \
                        --memory_size $EXEMPLARS \
                        --exem_training_batch_size $EXEM_TRAINING \
                        --fe_learning_rate $FE_LEARNING_RATE \
                        2>&1 | tee /data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/loss_decoupling/${DATASET}/log.log