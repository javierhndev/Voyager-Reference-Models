#!/bin/bash

############

set -ex

NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

#PYTHONPATH=$PYTHONPATH:$MODEL_ROOT/Pytorch/common:$MODEL_ROOT;
PYTHONPATH=$PYTHONPATH:$MODEL_ROOT/Pytorch/common;

# ----------------------
# Configurable parameters
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/llama/MDS_REFS/}
NUM_NODES=${NUM_NODES:-1}
DP=${HL_DP:-1}
TP=${HL_TP:-4}
PP=${HL_PP:-2}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=/output # ${results:-}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-$OMPI_MCA_orte_default_hostfile}
USE_HPU=${HL_USE_HPU:-1}
CKP_ACT=${HL_CKP_ACT:-0}
UNIV_CP=${HL_UNIV_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-1}
LLAMA_VER=${HL_LLAMA_VER:-13}
# ----------------------

# Dataset
ARXIV=${DATA_DIR}/arxiv/tokenized_text_document
BOOKS=${DATA_DIR}/book/tokenized_text_document
C4=${DATA_DIR}/c4/tokenized_text_document
GITHUB=${DATA_DIR}/github/tokenized_text_document
STACKEXC=${DATA_DIR}/stackexchange/tokenized_text_document
WIKI=${DATA_DIR}/wikipedia//tokenized_text_document
COMMON=${DATA_DIR}/common_crawl/combined/combined
DATA_PATH="2.5 ${ARXIV} 4.5 ${BOOKS} 15 ${C4} 4.5 ${GITHUB} 2.0 ${STACKEXC} 4.5 ${WIKI} 67 ${COMMON}"

#scaling
NUM_GPUs=$(($DP * $TP * $PP))

## Llama-7B model architecture
NLAYERS=32 # must be divisible by PP
NHIDDEN=4096
NHEADS=32 # must be divisible by TP
FFN_HIDDEN_SIZE=11008

SEQ_LEN="${SEQ_LEN:-2048}";

# Training parameters
GLOBAL_BATCH="${GLOBAL_BATCH:-256}";
ZERO_STAGE="${ZERO_STAGE:-0}";


if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# handle kill switch argument
if [ -z "$KILL_SWITCH_FILE"]; then
    KILL_SWITCH_ARG=""
else
    KILL_SWITCH_ARG="--kill-switch-path $KILL_SWITCH_FILE"
fi


# create DS config
DS_CONFIG=${OUTPUT_DIR}/ds_config.json
cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "reduce_scatter": false
  },
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false, 
  "checkpoint": {"tag_validation":  "IGNORE" }
}
EOT


#not sure if needed here
HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};

#configure deepspeed environment
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $HOME/.deepspeed_env;
echo "PYTHONPATH=/scratch/Model-References/PyTorch/common:$PYTHONPATH" >> $HOME/.deepspeed_env;
echo "PATH=$PATH" >> $HOME/.deepspeed_env;

cat $HOME/.deepspeed_env;



CMD="python -u $MODEL_PATH/pretrain_llama.py \
    --deepspeed \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --position-embedding-type rotary \
    --no-bias \
    --layernorm-type rmsnorm \
    --activation-func-type swiglu \
    --layernorm-epsilon 1e-6 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-iters 10000 \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 10 \
    --eval-interval 100 \
    --data-path ${DATA_PATH} \
    --vocab-file $DATA_DIR/gpt2-vocab.json \
    --merge-file $DATA_DIR/gpt2-merges.txt \
    --optimizer adamw \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-6 \
    --lr 3e-4 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load $CHECKPOINTS_DIR \
    --deepspeed_config=$DS_CONFIG  \
    --zero-stage=$ZERO_STAGE \
    --exit-interval $EXIT_INTERVAL \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-pipeline-parallel \
    $KILL_SWITCH_ARG \
    --bf16"

if [ $USE_HPU -eq 1 ]
then
    CMD="${CMD} --use_hpu --distributed-backend=hccl --hpu-deterministic"
fi

if [ $UNIV_CP -eq 1 ]
then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint --load $WORKER_DIR/llama13-uc/checkpoints"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]
then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL --verify-checkpoint --verify-checkpoint-model-type LLAMA"
fi

if [ $CKP_ACT -eq 1 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing"
fi


deepspeed --force_multi \
          --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          --hostfile=$HOSTSFILE \
          --master_addr=$MASTER_ADDR \
          $CMD;
