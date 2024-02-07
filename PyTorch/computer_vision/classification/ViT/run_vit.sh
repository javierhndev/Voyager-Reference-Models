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



CMD="python $MODEL_PATH/train.py \
	--name imagenet1k_TF \
	--dataset imagenet1K \
	--data_path /dataset \
	--model_type ViT-B_16 \
	--pretrained_dir /pretrained/ViT-B_16.npz \
	--output_dir /output \
	--num_steps 20000 \
	--eval_every 1000 \
	--train_batch_size 64 \
	--gradient_accumulation_steps 2 \
	--img_size 384 \
	--learning_rate 0.06 \
	--hmp --hmp-opt-level O1 --hmp-bf16 $MODEL_PATH/ops_bf16.txt --hmp-fp32 $MODEL_PATH/ops_fp32.txt";


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


