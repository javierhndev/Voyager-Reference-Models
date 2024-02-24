#!/bin/bash

############

set -ex




#HOME='/scratch/tmp';
NUM_NODES=${NUM_NODES:-1};
NGPU_PER_NODE=8;
N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

#MODEL_PATH=/scratch/Model-References/PyTorch/nlp/bloom;
PYTHONPATH=$PYTHONPATH:$MODEL_ROOT;
#PATH=/scratch/tmp/.local/bin:$PATH;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
MASTER_PORT=${MASTER_PORT:-15566};

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $HOME/.deepspeed_env;
echo "PYTHONPATH=/scratch/Model-References/PyTorch/common:$PYTHONPATH" >> $HOME/.deepspeed_env;
echo "PATH=$PATH" >> $HOME/.deepspeed_env;

echo $MASTER_ADDR;
echo $MASTER_PORT;

cat $HOME/.deepspeed_env;

#ds_report;

##
MODEL_BLOOM=bloom-7b1;
#big one MODEL_BLOOM=bloom;

CMD="python $MODEL_PATH/bloom.py \
     --weights /dataset \
     --model ${MODEL_BLOOM} \
     --options "max_length=128" \
     --dtype bf16 \
     'Do robots think?' ";

deepspeed --force_multi \
          --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          --hostfile=$HOSTSFILE \
          --master_addr=$MASTER_ADDR \
          $CMD;
