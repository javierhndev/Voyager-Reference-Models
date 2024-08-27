# ResNet50 for Pytorch Lightning
(**NOTE**: The model is no longer supported by Intel Habana in their last SynapaseAI versions but we have verified that it works on Voyager)
Here we provide the yaml files and instructions to train the ResNet50 model, using PyTorch Lightning, on Voyager.


## Overview

The model was supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/classification/lightning/resnet). This tutorial uses SynapseAI v1.15.1 (but the model was written for 1.13).



## DATASET

### Prepare the dataset
The ResNet50 model uses the ImageNet2015 dataset. The steps to obtain it are the same as for [Pytorch ResNet50 model](/PyTorch/computer_vision/classification/torchvision).

**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012
```



## TRAINING

### Single-card

A single Gaudi card is enough to test the model. You can find in this folder the `resnet_light_1card.yaml` file to run the model. Remember to modify the `dataset` environment to the location of your Imagenet2015 dataset.

This example will execute the model with:
```bash
python3 resnet50_PTL.py --batch_size 256\
                     --data_path ${dataset} \
                     --autocast \
                     --custom_lr_values 0.1 0.01 0.001 0.0001 \
                     --custom_lr_milestones 0 30 60 80 \
                     --hpus 1 \
                     --max_train_batches 500 \
                     --epochs 2 ;
```


### Multi-Card Training

**Run training on 8 HPUs:**

The execution of this model on multiple cards is slighyly different than other Pytorch models. Here we execute the mpirun `-npernode 1` and then Pythorch Lightning handles the parallelization itself.

To run on multiple cards we launch an MPI job with three files: `resnet_light_8cards.yaml`, `setup.sh` and `run_resnet.sh`. Some variables in the `resnet_light_8cards.yaml` need to be modified to your own configuration:
- `RUN_PATH`: The location where you keep those three files.
- `dataset`: The location of the ILSDVRC2012 dataset.


Neither `setup.sh` or `run_resnet.sh` need to be modified. Once the MPI job is launched, the `run_resnet.sh` will execute the following:
```bash
python3 $MODEL_PATH/resnet50_PTL.py \
             --batch_size 256 \
             --data_path ${dataset} \
             --autocast \
             --epochs 5 \
             --custom_lr_values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 \
             --custom_lr_milestones 1 2 3 4 30 60 80 \
             --hpus $N_CARDS \
             --max_train_batches 500

```





