# Reference Models for Voyager
This repository contains the necessary files and links to run a collection of models on Voyager at the San Diego Supercomputer Center (SDSC). The majority of those models are the ones supported by Intel Habana [link](https://github.com/HabanaAI/Model-References).

The list of models is being improved. Your feedback is greatly appreciated. Feel free to open an issue or contact me (Javier Hernandez-Nicolau) at `javierhn *at* ucsd.edu`.

## Model list

## Simple examples
- ['Hello World!'](helloworld): simplest example to run on Voyager.
- [Fashion-MNIST](TensorFlow/examples/Fashion-MNIST) model. It shows how to run a python script with TensorFlow.
- [MPIJob](TensorFlow/examples/MPIJob) Learn how to run a MNIST model in multiple HPUs. 

## Computer Vision
| Models                                                                                 | Framework  | Multi-node | Verified |
| -------------------------------------------------------------------------------------- | ---------- | ---------- | -------- |
| [ResNet50, ResNet152, ResNeXt101](PyTorch/computer_vision/classification/torchvision)  | Pytorch    |  Yes       |  1.11    |
| [ResNet50 (Pytorch Lightning](PyTorch/computer_vision/classification/lightning/resnet) | Pytorch    |            |  1.13    |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                      | Pytorch    |  Yes       |  1.11    |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                        | Pytorch    |  Yes       |  1.11    |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                                       | Pytorch    |  Yes       |  1.11    |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)                       | Pytorch    |  Yes       |  1.11    |
| [UNet-2D](TensorFlow/computer_vision/Unet2D)                                           | TensorFlow |            |  1.11    |
| [ResNet50 (Keras)](TensorFlow/computer_vision/Resnets/resnet_keras)                    | TensorFlow |  Yes       |  1.11    |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt)                               | TensorFlow |  Yes       |  1.11    |


## Natural Language processing
| Models                                                                        | Framework  | Multi-node  | Verified |
| ----------------------------------------------------------------------------- | ---------- | ----------- | -------- |
| [BERT](PyTorch/nlp/bert)                                                      | Pytorch    |  Yes        |   1.11   |
| [BART (fine-tuning)](PyTorch/nlp/BART)                                        | Pytorch    |             |   1.11   |
| [Hugginface BLOOM (inference)](PyTorch/nlp/bloom)                             | Pytorch    |             |   1.11   |
| [LLaMA (Megatron-DeepSpeed](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) | Pytorch    |  Yes        |   1.11   |
| [BERT](TensorFlow/nlp/bert)                                                   | TensorFlow |             |   1.11   |

## Generative models
| Models                                                         | Framework  | Multi-node | Verified |
| -------------------------------------------------------------- | ---------- | ---------- | -------- |
| [Stable-Diffusion](PyTorch/generative_models/stable-diffusion) | Pytorch    |            |   1.11   |
