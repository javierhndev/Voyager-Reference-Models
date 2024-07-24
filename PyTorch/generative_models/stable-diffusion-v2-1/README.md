# Inference with Stable Diffusion 2.1 for Pytorch
Here we provide the yaml files and instructions to run inference on Stable Diffusion 2.1 on Voyager.

## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.15.1/PyTorch/generative_models/stable-diffusion-v-2-1). This tutorial uses SynapseAI v1.15.1.

This implementation is based on the following paper - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder [(OpenCLIP-ViT/H)](https://github.com/mlfoundations/open_clip). The [model](https://github.com/Stability-AI/stablediffusion/) uses [Hugginface](https://huggingface.co/stabilityai/stable-diffusion-2-1) transformers. 


## DATASET

### Text-to-Image
Download the pre-trained weights for 768x768 images (4.9GB)
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
```

**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/stable_pretrained
```

Feel free to use it!

## TRAINING

Single Gaudi cards have not enough memory to run the model so it needs to be run on multiple cards.

### Multi-Card Training

**Run training on 8 HPUs:**

To run on multiple cards we launch an MPI job with three files: `sd_8cards.yaml`, `setup.sh` and `run_sd.sh`. Some variables in the `yolox_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `pretrained`: The location of the checkpoint.
- `output`: Output files will be written on this folder.  

Remember to change them in the launcher **and** worker.

Neither `setup.sh` or `run_sd.sh` need to be modified to run but your prompt must be specified on `run_sd.sh`. Once the MPI job is launched, the `run_sd.sh` will execute the following:
```bash
python $MODEL_PATH/scripts/txt2img.py --ckpt /pretrained/v2-1_768-ema-pruned.ckpt --config $MODEL_PATH/configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --n_samples 1 --n_iter 5 --outdir /output --prompt 'a cartoon style picture of a supercomputer';
```


