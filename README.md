# Reference Models for Voyager
This repository contains the necessary files and links to run a collection of models on Voyager at the San Diego Supercomputer Center (SDSC). The majority of those models are the ones supported by Intel Habana [link](https://github.com/HabanaAI/Model-References).

The list of models is being improved. Your feedback is greatly appreciated. Feel free to open an issue or contact me (Javier Hernandez-Nicolau) at `javierhn *at* ucsd.edu`.

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
| [YOLOX](PyTorch/computer_vision/detection/yolox)                                      | Pytorch    |  Yes       |
| [UNet-2D](TensorFlow/computer_vision/Unet2D)                                          | TensorFlow |            |
| [ResNet50 (Keras)](TensorFlow/computer_vision/Resnets/resnet_keras)                   | TensorFlow |  Yes       |


## Natural Language processing
| Models                                                                        | Framework  | Multi-node |
| ----------------------------------------------------------------------------- | ---------- | ---------- |
| [BERT](PyTorch/nlp/bert)                                                      | Pytorch    |  Yes       |
| [BART (fine-tuning)](PyTorch/nlp/BART)                                        | Pytorch    |            |
| [Hugginface BLOOM (inference)](PyTorch/nlp/bloom)                             | Pytorch    |            |
| [LLaMA (Megatron-DeepSpeed](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) | Pytorch    |  Yes       |
| [BERT](TensorFlow/nlp/bert)                                                   | TensorFlow |            |

## Generative models
| Models                                                         | Framework  | Multi-node |
| -------------------------------------------------------------- | ---------- | ---------- |
| [Stable-Diffusion](PyTorch/generative_models/stable-diffusion) | Pytorch    |            |
