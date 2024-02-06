#!/bin/sh

cd /scratch;

git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

pip install -r $MODEL_PATH/requirements.txt;

pip install -v -e;
