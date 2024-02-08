#!/bin/bash

############

set -ex


NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

PYTHONPATH=$PYTHONPATH:$MODEL_ROOT;


CMD="python3 $MODEL_PATH/imagenet_main.py \
	--use_horovod \
	-dt bf16 \
	-dlit fp32 \
	-bs 128 \
	-te 90 \
	-ebe 90 \
	--data_dir /dataset";


mpirun -np ${N_CARDS} \
       --allow-run-as-root \
       --bind-to core \
       --map-by ppr:4:socket:PE=6 \
       -rank-by core --report-bindings \
       --tag-output \
       --output-filename /output/resnext_log \
       --merge-stderr-to-stdout --prefix $MPI_ROOT \
       -x MASTER_ADDR=$MASTER_ADDR \
       -x MASTER_PORT=$MASTER_PORT \
       -x PYTHONPATH \
       $CMD;


