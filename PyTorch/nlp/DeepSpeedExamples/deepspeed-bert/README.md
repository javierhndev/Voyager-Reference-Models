# DeepSpeed BERT-1.5B and BERT-5B for PyTorch
This folder contains the scripts and yaml files to launch BERT-1.5B and BERT-5B with Deepspeed on Voyager.

## Model overview

Bert is a technique for natural language processing (NLP) pre-training developed by Google. The models here has been ported by [Intel Habana](https://github.com/HabanaAI/Model-References/tree/1.15.1/PyTorch/nlp/DeepSpeedExamples/deepspeed-bert) and is based on a clone of [https://github.com/NVIDIA/DeepLearningExamples.git](https://github.com/NVIDIA/DeepLearningExamples.git). The model has been verified on SynapseAI 1.15.1.

In this pre-training model script we will introduce the following models:

- **BERT-1.5B**: 48-layer, 1600-hidden, 25-heads, 1.5B parameters neural network architecture.
- **BERT-5B**: a 63-layer, 2560-hidden, 40-heads, 5B parameters neural network architecture.

### BERT-1.5B and BERT-5B Pre-Training
BERT-1.5B and BERT-5B pre-training with DeepSpeed library includes the following configurations:

- Multi-card data parallel with Zero1 and BF16 (BERT-1.5B).
- Multi-server data parallel with Zero1 and BF16 (BERT-1.5B and BERT-5B).
- Dataset: wiki.
- BERT-5B uses optimizer: LANS
- Consists of two tasks:
  - Task 1 - Masked Language Model - where when given a sentence, a randomly chosen word is guessed.
  - Task 2 - Next Sentence Prediction - where the model guesses whether sentence B comes after sentence A.

## Dataset
Please follow the same instructions as in [BERT for Python](/PyTorch/nlp/bert). Right now, a copy is located at:
```bash
/voyager/ceph/users/javierhn/datasets/bert/pytorch
```

## Pre-Training the models

We provide the yaml and scripts to run the BERT-1.5B and BERT-5B in one or multiple nodes.

### BERT-1.5B
To run **BERT-1.5B** in a single node, launch the `bert_deepspeed_1.5b_8x.yaml`. In this yaml, you need to specify the `mydir` (where the scipts are located) and `dataset` (wiki dataset location) volumes. The yamls will execute the `setup.sh` on the node which will install the requirements and Deepspeed. Then, it will launch the `run_bert_1.5b.sh` (where you can tune some parameters).

To run on multiple nodes use: `bert_deepspeed_1.5b_16x.yaml` or `bert_deepspeed_1.5b_32x.yaml`.

### BERT-5B
Similaryly, the **BERT-5B** can be run on 4 nodes (32 HPU cards) launching `bert_deepspeed_5b_32x.yaml`. 
