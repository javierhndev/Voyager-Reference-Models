#!/bin/bash

############

set -ex


NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

PYTHONPATH=$PYTHONPATH:$MODEL_ROOT;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};

echo $MASTER_ADDR;
echo $MASTER_PORT;



CMD="python $MODEL_PATH/ssd/train.py \
	-d /dataset \
	--batch-size 128 \
	--log-interval 100 \
	--val-interval 10 \
	--use-hpu \
	--hpu-lazy-mode \
	--autocast \
	--warmup 2.619685 \
	--num-workers 12";
#	--pretrained-backbone /pretrained/resnet34-333f7ec4.pth \
	
mpirun -np ${N_CARDS} \
       --allow-run-as-root \
       --bind-to core \
       --map-by ppr:4:socket:PE=6 \
       -rank-by core --report-bindings \
       --tag-output \
       --merge-stderr-to-stdout --prefix $MPI_ROOT \
       -x MASTER_ADDR=$MASTER_ADDR \
       -x MASTER_PORT=$MASTER_PORT \
       -x PYTHONPATH \
       $CMD;


