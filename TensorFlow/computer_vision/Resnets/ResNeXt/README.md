# ResNeXt for TensorFlow
Here we provide the yaml files and instructions to train the ResNeXt model (TensorFlow) on Voyager.


## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.11.0/TensorFlow/computer_vision/Resnets/ResNeXt). This tutorial uses SynapseAI v1.11.

ResNeXt is a modified version of the original ResNet v1 model. This implementation defines ResNeXt101 which features 101 layers.







## DATASET

The ResNeXt model uses the ImageNet2015 dataset from thje ILSVRC challenge. The steps to obtain it are the same as for [TensorFlow ResNet50 model](/TensorFlow/computer_vision/Resnets/resnet_keras).

**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012/tf_records
```



## TRAINING

### Single Card training
A single card pod can be run using the `resnext_1card.yaml` file in this folder. You need to modify the `\dataset` lcoation the yaml file to match the one where you have the ILSVRC2012 dataset.

The pod will execute the followin:
```bash
python3 imagenet_main.py -dt bf16 -dlit fp32 -bs 128 -te 90 -ebe 90 --data_dir /dataset
```
wich will run the ResNeXt model with 1 HPU, BF16, batch 128 and 90 epochs.


### Multi-Card Training

**Run training on 8 HPUs (Horovod):**

To run on multiple cards we launch an MPI job with three files: `resnext_8cards.yaml`, `setup.sh` and `run_resnext.sh`. Some variables in the `resnext_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of the ILSDVRC2012 dataset.
- `output`: Output files will be written on this folder.  

Remember to change them in the launcher **and** worker.

Neither `setup.sh` or `run_resnext.sh` need to be modified (unless you want to modify the model). Once the MPI job is launched, the `run_vit.sh` will execute the following:
```bash
python3 $MODEL_PATH/imagenet_main.py \
        --use_horovod \
        -dt bf16 \
        -dlit fp32 \
        -bs 128 \
        -te 90 \
        -ebe 90 \
        --data_dir /dataset
```
The `ResNeXt` model will run on 8 HPU with BF16, batch 128, 90 epochs.

**Run training on 16 HPUs (or more):**

To run on multiple nodes you can use the `resnext_16cards.yaml` job file. In this example we run on two nodes (16 cards) and the only difference from the `resnext_8cards.yaml` is on `NUM_NODES=2` and `replicas=2` (in the workers).



