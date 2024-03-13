#!/bin/bash

############

set -ex




#HOME='/scratch/tmp';
NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

PYTHONPATH=$PYTHONPATH:$MODEL_ROOT:$MODEL_PATH;
#PATH=/scratch/tmp/.local/bin:$PATH;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};

echo $MASTER_ADDR;
echo $MASTER_PORT;


#CMD="python $MODEL_PATH/scripts/txt2img.py --ckpt /pretrained/v2-1_768-ema-pruned.ckpt --config $MODEL_PATH/configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --n_samples 1 --n_iter 3 --outdir /output --prompt 'a horse'" ;


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
       python $MODEL_PATH/scripts/txt2img.py --ckpt /pretrained/v2-1_768-ema-pruned.ckpt --config $MODEL_PATH/configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --n_samples 1 --n_iter 5 --outdir /output --prompt 'a cartoon style picture of a supercomputer';
       #$CMD;


