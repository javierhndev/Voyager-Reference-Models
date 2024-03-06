#!/bin/bash

############

#set -ex


if [ -z ${OMPI_COMM_WORLD_SIZE} ];
then WORLD_SIZE=1;
else WORLD_SIZE=${OMPI_COMM_WORLD_SIZE};
fi;

if [ -z ${OMPI_COMM_WORLD_RANK} ];
then NODE_RANK=0;
else NODE_RANK=${OMPI_COMM_WORLD_RANK};
fi;

export WORLD_SIZE=${WORLD_SIZE}
export NODE_RANK=${NODE_RANK}

#NODE_RANK=0;

echo $WORLD_SIZE;
echo $NODE_RANK;
echo $MASTER_ADDR;
echo $MASTER_PORT;

#unset $(env | grep "OMPI_" | cut -d= -f1); 

#unset $(env | grep "PMIX_" | cut -d= -f1);

#unset the MPI env variables so PTLightning can do the parallelization
for v in $(printenv |grep OMPI | cut -d '=' -f 1); do
  unset $v
done


CMD="python3 -u $MODEL_PATH/main.py \
                      --results /output \
                      --task 1 \
                      --logname res_log \
                      --fold 0 \
                      --hpus $N_CARDS \
                      --gpus 0 \
                      --data /dataset \
                      --seed 123 \
                      --num_workers 8 \
                      --affinity disabled \
                      --norm instance \
                      --dim 2 \
                      --optimizer fusedadamw  \
                      --exec_mode train \
                      --learning_rate 0.001 \
                      --autocast \
                      --deep_supervision \
                      --batch_size 64 \
                      --val_batch_size 64 \
                      --min_epochs 30 \
                      --max_epochs 100 \
                      --train_batches 0 \
                      --test_batches 0";

$CMD;
