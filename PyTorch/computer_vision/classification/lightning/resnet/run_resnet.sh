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


CMD="python3 $MODEL_PATH/resnet50_PTL.py \
	     --batch_size 256 \
             --data_path ${dataset} \
	     --autocast \
             --epochs 5 \
             --custom_lr_values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 \
             --custom_lr_milestones 1 2 3 4 30 60 80 \
	     --hpus $N_CARDS \
	     --max_train_batches 500";

$CMD;
