# Vision Transformer for Pytorch
Here we provide the yaml files and instructions to train the Vision Transformer model on Voyager.

## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/classification/ViT). This tutorial uses SynapseAI v1.15.1.

This is a PyTorch implementation of the Vision Transformer model described in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) paper. It is based on an earlier implementation from [PyTorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models) and the [official repository](https://github.com/google-research/vision_transformer).

![fig1](https://github.com/HabanaAI/Model-References/blob/1.13.0/PyTorch/computer_vision/classification/ViT/img/figure1.png)

The Vision Transformer model achieves State-of-the-Art in image recognition task with the standard Transformer encoder and fixed-size patches. To perform classification, you can use the standard approach of adding an extra learnable "classification token" to the sequence.


## DATASET

### Download the pre-trained model

Google offers several pretrained models but, in this repository, we have tested the ViT-B_16(85.8M). For other supported models check [Habana's repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/classification/ViT).

To download the ViT-B_16 model (imagenet21k pre-train):
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
### Prepare the dataset
The Vision Transformers model uses the ImageNet2015 dataset. The steps to obtain it are the same as for [Pytorch ResNet50 model](/PyTorch/computer_vision/classification/torchvision).

**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012
```



## TRAINING

Single Gaudi cards have not enough memory to run the model so it needs to be run on multiple cards.

### Multi-Card Training

**Run training on 8 HPUs:**

To run on multiple cards we launch an MPI job with three files: `vit_8cards.yaml`, `setup.sh` and `run_vit.sh`. Some variables in the `vit_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of the ILSDVRC2012 dataset.
- `pretrained`: The location of the pre-trained model (the *.npz* file)
- `output`: Output files will be written on this folder.  

Remember to change them in the launcher **and** worker.

Neither `setup.sh` or `run_vit.sh` need to be modified. Once the MPI job is launched, the `run_vit.sh` will execute the following:
```bash
python $MODEL_PATH/train.py \
        --name imagenet1k_TF \
        --dataset imagenet1K \
        --data_path /dataset \
        --model_type ViT-B_16 \
        --pretrained_dir /pretrained/ViT-B_16.npz \
        --output_dir /output \
        --num_steps 20000 \
        --eval_every 1000 \
        --train_batch_size 64 \
        --gradient_accumulation_steps 2 \
        --img_size 384 \
        --learning_rate 0.06 \
        --autocast
```
The `ViT-B_16` model runs with batch size 64, gradient accumulation 2.

**Run training on 16 HPUs (or more):**

To run on multiple nodes you can use the `vit_16cards.yaml` job file. In this example we run on two nodes (16 cards) and the only difference from the `vit_8cards.yaml` is on `NUM_NODES=2` and `replicas=2` (in the workers).



