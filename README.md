# Reference Models for Voyager
This repository contains the necessary files or links to run a collection of models on Voyager at San Diego Supercomputer Center (SDSC). The majority of those models are the ones supported by Habana Labs [link](https://github.com/HabanaAI/Model-References).

## Model list

## Simple examples
- ['Hello World!'](helloworld): simplest example to run in Voyager.
- [Fashion-MNIST](TensorFlow/examples/Fashion-MNIST) model. It shows how to run a python script with TensorFlow.
- [MPIJob](TensorFlow/examples/MPIJob) Learn how to run a MNIST model in multiple HPUs. 

## Computer Vision
| Models                                                                                | Framework  | Multi-node |
| ------------------------------------------------------------------------------------- | ---------- | ---------- |
| [ResNet50, ResNet152, ResNeXt101](PyTorch/computer_vision/classification/torchvision) | Pytorch    |  Yes       |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                     | Pytorch    |  Yes       |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                       | Pytorch    |  Yes       |
| [UNet-2D](TensorFlow/computer_vision/Unet2D)                                          | TensorFlow |            |
| [ResNet50 (Keras)](TensorFlow/computer_vision/Resnets/resnet_keras)                   | TensorFlow |  Yes       |


## Natural Language processing
| Models                                 | Framework  | Multi-node |
| -------------------------------------- | ---------- | ---------- |
| [BART (fine-tuning)](PyTorch/nlp/BART) | Pytorch    |            |
| [BERT](TensorFlow/nlp/bert)            | TensorFlow |            |

## Generative models
| Models                                                         | Framework  | Multi-node |
| -------------------------------------------------------------- | ---------- | ---------- |
| [Stable-Diffusion](PyTorch/generative_models/stable-diffusion) | Pytorch    |            |
