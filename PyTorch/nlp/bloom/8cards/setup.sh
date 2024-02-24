#!/bin/sh

hl-smi;

cd /scratch;

#mkdir -p /scratch/tmp;

git clone -b $SYNAPSE_AI_VER https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

pip install -r $MODEL_PATH/requirements.txt;

pip install git+https://github.com/HabanaAI/DeepSpeed.git@$SYNAPSE_AI_VER;
ds_report;

apt update
apt install -y pdsh
