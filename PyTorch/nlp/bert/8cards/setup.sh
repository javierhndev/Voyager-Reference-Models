#!/bin/sh

cd /scratch;

mkdir -p /scratch/tmp;

git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;

cd $MODEL_PATH;

pip install -r $MODEL_PATH/requirements.txt;

pip install h5py==3.10.0 boto3==1.26.75 git+https://github.com/NVIDIA/dllogger.git@26a0f8f1958de2c0c460925ff6102a4d2486d6cc;

