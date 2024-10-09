# Reference Models for Voyager
This repository contains the necessary files and links to run a collection of models on Voyager at the San Diego Supercomputer Center (SDSC). The majority of those models are the ones supported by Intel Habana [link](https://github.com/HabanaAI/Model-References).

The last column on the list (*Verified*) indicates the last version of Synapse AI the model was tested on Voyager.

The list of models is being improved. Your feedback is greatly appreciated. Feel free to open an issue or contact me (Javier Hernandez-Nicolau) at `javierhn *at* ucsd.edu`.

## Model list

## Simple examples
- ['Hello World!'](helloworld): simplest example to run on Voyager.
- [Port your Pytorch application](/PyTorch/examples/port_mnist). An example of a CNN model ported to Intel Gaudi HPU card.
- [MPIJob TF](TensorFlow/examples/MPIJob) Learn how to run a MNIST Tensorflow model in multiple HPUs. 
- [MPIJob PT](PyTorch/examples/MPIJob/) Learn how to run a MNIST Pytorch model in multiple HPUs.
- [Tutorial: Huggingface Diffusers](PyTorch/examples/huggingface/diffusers) Learn ho to run Stable Diffusion Inference on HPUs.

## Computer Vision
| Models                                                                                  | Framework         | Multi-node | Verified |
| --------------------------------------------------------------------------------------- | ----------------- | ---------- | -------- |
| [ResNet50, ResNet152, ResNeXt101](PyTorch/computer_vision/classification/torchvision)   | Pytorch           |  Yes       |  1.15.1  |
| [ResNet50 (Pytorch Lightning)](PyTorch/computer_vision/classification/lightning/resnet) | Pytorch Lightning |            |  1.15.1  |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                       | Pytorch           |  Yes       |  1.15.1  |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                         | Pytorch           |  Yes       |  1.15.1  |
| [UNet2D,UNet3D](PyTorch/computer_vision/segmentation/unet)                              | Pytorch Lightning |            |  1.15.1  |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                                        | Pytorch           |  Yes       |  1.15.1  |
| [SSD](PyTorch/computer_vision/detection/ssd)                                            | Pytorch           |            |  1.15.1  |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)                        | Pytorch           |  Yes       |  1.15.1  |
| [UNet-2D](TensorFlow/computer_vision/Unet2D)                                            | TensorFlow        |  Yes       |  1.15.1  |
| [ResNet50 (Keras)](TensorFlow/computer_vision/Resnets/resnet_keras)                     | TensorFlow        |  Yes       |  1.15.1  |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt)                                | TensorFlow        |  Yes       |  1.15.1  |


## Natural Language processing
| Models                                                                          | Framework  | Multi-node  | Verified |
| ------------------------------------------------------------------------------- | ---------- | ----------- | -------- |
| [BERT](PyTorch/nlp/bert)                                                        | Pytorch    |  Yes        |   1.15.1 |
| [BART (fine-tuning,simpletransformers)](PyTorch/nlp/BART)                       | Pytorch    |  Yes        |   1.15.1 |
| [Hugginface BLOOM (inference)](PyTorch/nlp/bloom)                               | Pytorch    |             |   1.15.1 |
| [BERT 1.5B and 5B with Deepspeed](PyTorch/nlp/DeepSpeedExamples/deepspeed-bert) | Pytorch    |  Yes        |   1.15.1 |
| [LLaMA (Megatron-DeepSpeed)](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed)  | Pytorch    |  Yes        |   1.13   |
| [BERT](TensorFlow/nlp/bert)                                                     | TensorFlow |             |   1.13   |

## Generative models
| Models                                                                                           | Framework  | Multi-node | Verified |
| ------------------------------------------------------------------------------------------------ | ---------- | ---------- | -------- |
| [(Not supported now) Stable-Diffusion](PyTorch/generative_models/stable-diffusion)               | Pytorch    |            |   1.11   | 
| [Stable-Diffusion 2.1 (Inference, Huggingface)](PyTorch/generative_models/stable-diffusion-v2-1) | Pytorch    |            |   1.15.1 |
