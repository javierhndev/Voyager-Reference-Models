#!/bin/sh

cd /scratch;

git clone -b $SYNAPSE_AI_VER https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

pip install -r $MODEL_PATH/requirements.txt;
