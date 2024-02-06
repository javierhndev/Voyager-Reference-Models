#!/bin/bash

############

set -ex


NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

PYTHONPATH=$PYTHONPATH:$MODEL_PATH;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};

echo $MASTER_ADDR;
echo $MASTER_PORT;



CMD="python $MODEL_PATH/tools/train.py \
           --name yolox-s \
	   --devices $N_CARDS \
	   --batch-size 128 \
	   --data_dir /dataset \
	   --hpu max_epoch 2 output_dir /output";


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


