# Reference Models for Voyager
This repository contains the necessary files or links to run a collection of models on Voyager at San Diego Supercomputer Center (SDSC). The majority of those models are the ones supported by Habana Labs [link](https://github.com/HabanaAI/Model-References).

## Model list

## Simple examples
- ['Hello World!'](helloworld): simplest example to run in Voyager.
- [Fashion-MNIST](TensorFlow/examples/Fashion-MNIST) model. It shows how to run a python script with TensorFlow.
- [MPIJob](TensorFlow/examples/MPIJob) (Working on it) 

## Computer Vision
| Models                                                                                | Framework  |
| ------------------------------------------------------------------------------------- | ---------- |
| [ResNet50, ResNet152, ResNeXt101](PyTorch/computer_vision/classification/torchvision) | Pytorch    |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                     | Pytorch    |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                       | Pytorch    |


## Natural Language processing
| Models                      | Framework  |
| --------------------------- | ---------- |
| [BERT](TensorFlow/nlp/bert) | TensorFlow |

## Generative models
| Models                                                         | Framework  |
| -------------------------------------------------------------- | ---------- |
| [Stable-Diffusion](PyTorch/generative_models/stable-diffusion) | Pytorch    |
