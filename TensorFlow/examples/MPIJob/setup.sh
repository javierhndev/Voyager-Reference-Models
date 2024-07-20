#!/bin/sh

hl-smi;

cd /scratch;

#mkdir -p /scratch/tmp;

git clone -b $SYNAPSE_AI_VER https://github.com/HabanaAI/Model-References;

