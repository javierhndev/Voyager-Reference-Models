#!/bin/sh

cd /scratch;

mkdir -p /scratch/tmp;

git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

pip install -r $MODEL_PATH/requirements.txt;

pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.11.0;
#ds_report;

apt update
apt install -y pdsh
