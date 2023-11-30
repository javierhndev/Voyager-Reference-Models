# Computer vision classification with Pytorch
Here we provide the yaml files and instructions to train ResNet50, ResNet152, ResNeXt101, MobbileNetV2 & GoogLeNet models on Voyager.

## Overview

The models are supported by Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision). This tutorial uses SynapseAI v1.11. The base model used here is actually from [GitHub: PyTorch-Vision](https://github.com/pytorch/vision/tree/release/0.10/torchvision/models) which has been modifed by Habana Labs. The following models have been tested on Voyager:

- ResNet50
- ResNet152
- ResNext101
- MobileNetV2
- GoogLeNet


## DATASET

All the models use the [ImageNet 2012 dataset](https://image-net.org/download.php). To download the data you need to create an account and accept the terms of access. Our models require the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files. Once you have downloaded the files, use the following commands (more details on [https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)):

```bash
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```  
Note that the dataset is ~170Gb and takes several hours to download

** Alternative **
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012
```

Feel free to use it!

## TRAINING

We are showing here some examples for each model. You can find more examples with differen parameters in [Habana's repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision)
- To see the available training parameters for ResNet50, ResNet152, ResNeXt101 and MobileNetV2, run:
  ```bash
  python3 -u train.py --help
  ```
- To see the available training parameters for GoogLeNet, run:
  ```bash
  python3 -u main.py --help
  ```

All the yaml files here use two environment variables: `dataset` and `output`. Use them to point the location of the dataset and output folder (Use Ceph). Multi-card examples also define a path for the Habana's `Model-References` and a `RUN_PATH` (where the `.yaml` file and `setup.sh` are located).
 
Note that the number of epochs for every run has been set to 1 for testing.

### Training examples (single card and multi-card)
** Run on 1 HPU **
You can find the yaml files in the `1card` folder. Execute them with
```bash
kubectl create -f themodel.yaml
```
to launch a pod to run the model. The code is dowloaded in to the scratch. The location of the dataset of the model is defined in `dataset` variable. You can also choose your output folder with the `output` environment variable. 

Note each run takes ~3 hours to run 1 epoch in 1 HPU.

- ResNet50 (lazy mode, BF16 mixed precision, batch Size 256, custom learning rate, Habana dataloader):
  ```bash
  kubectl create -f resnet50_1card.yaml 
  ```
  which will execute the following:
  ```bash
  python3 -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path ${dataset} --output-dir ${output} --save-checkpoint --epochs 1 --autocast  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80 
  ```
- ResNeXt101 (lazy mode, BF16 mixed precision, batch size 256, custom learning rate, Habana dataloader):
  ```bash
  kubectl create -f resnext101_1card.yaml 
  ```
  which will execute the following:
  ```bash
  python3 -u train.py --dl-worker-type HABANA --batch-size 256 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path ${dataset} --output-dir ${output} --save-checkpoint --epochs 1 --autocast --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```

- ResNet152 (lazy mode, BF16 mixed precision, batch size 128, custom learning rate):
  ```bash
  kubectl create -f resnet152_1card.yaml
  ```
  which will execute the following:
  ```bash
  python3 -u train.py --dl-worker-type HABANA --batch-size 128 --model resnet152 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path ${dataset} --output-dir ${output} --save-checkpoint --epochs 1 --autocast --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- MobileNetV2 (lazy mode, BF16 mixed precision, batch size 256, 1 HPU on a single server with default PyTorch dataloader):
  ```bash
  kubectl create -f mobilenetv2_1card.yaml
  ```
  which will execute the following:
  ```bash
  python3 -u train.py --batch-size 256 --model mobilenet_v2 --device hpu --print-freq 10 --deterministic --data-path ${dataset} --output-dir ${output} --save-checkpoint --epochs 1 --autocast --dl-time-exclude=False --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --momentum 0.9 
  ```
- GoogLeNet (batch size 128, FP32 precision, lazy mode):
  ```bash
  kubectl create -f googlenet_1card.yaml
  ```
  which will execute the following:
  ```bash
  python3 -u main.py --batch-size 128 --data-path ${dataset} --device hpu --dl-worker-type HABANA --epochs 1 --lr 0.07 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
  ```


** Run on 8 HPUs **


** Run on 16 or 32 HPUs **


