#!/bin/sh

cd /scratch;

mkdir -p /scratch/tmp;

git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

python3 -m pip install -r $MODEL_PATH/requirements.txt;
