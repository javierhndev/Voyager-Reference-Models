#!/bin/bash

#####################################################################################
# Example: Pretraining phase 1 of BERT with 1.5B parameters on multicard i.e 8 cards
#####################################################################################
NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};



# Params: run_pretraining
DATA_DIR=/dataset/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en
MODEL_CONFIG=$MODEL_PATH/scripts/bert_1.5b_config.json
DS_CONFIG=$MODEL_PATH/scripts/deepspeed_config_bert_1.5b.json
RESULTS_DIR=$MODEL_PATH/results/bert_1.5b
MAX_SEQ_LENGTH=128
NUM_STEPS_PER_CP=200
MAX_STEPS=155000
RUN_STEPS=-1
LR=0.0015
WARMUP=0.05
CONST=0.25
LOG_FREQ=10
MAX_PRED=20
# Params: DeepSpeed
#NUM_NODES=1
#NGPU_PER_NODE=8

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

CMD="python -u $MODEL_PATH/run_pretraining.py \
     --disable_progress_bar \
     --optimizer=lans \
     --use_lr_scheduler \
     --resume_from_checkpoint \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=$MAX_PRED \
     --max_steps=$MAX_STEPS \
     --steps_this_run=$RUN_STEPS \
     --num_steps_per_checkpoint=$NUM_STEPS_PER_CP \
     --learning_rate=$LR \
     --warmup_proportion=$WARMUP \
     --constant_proportion=$CONST \
     --scheduler_degree=1.0 \
     --log_freq=$LOG_FREQ \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

mkdir -p $RESULTS_DIR

deepspeed --force_multi \
          --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          --hostfile=$HOSTSFILE \
          --master_addr=$MASTER_ADDR \
          $CMD;
