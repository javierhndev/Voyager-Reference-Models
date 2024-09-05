# HuggingFace diffusers on Voyager
Here we show how to run HuggingFace difffusers on Voyager using [optimum-habana](https://github.com/huggingface/optimum-habana/tree/main). As an example we will run inference with a model based on Stable Diffusion XL.

## Overview

[Huggingface diffusers](https://huggingface.co/docs/diffusers/index) allows you, in a few lines, to load a pretrained model and run inference. In this tutorial, we will run the [CommonCanvas-Xl](https://huggingface.co/common-canvas/CommonCanvas-XL-C) model which is based on the Stable Diffusion XL architecture.

[Optimum-habana](https://github.com/huggingface/optimum-habana/tree/main) is the interface between Huggingface Diffusers (and Transformers) and the Intel Gaudi accelerator (HPU).

This tutorial uses synapse AI 1.15.1 and optimum-habana 1.13.1

## Dataset

In our example we run a (local) model that has been previously downloaded from the Huggingface Hub. However, with huggingFace diffusers, you should be able to run any of their models in their repo.

To download any of the HF models you can use `git-lfs`. It is not available by default on Voyager but you can install it in a pod and run it from there (see the example on `dataset_download_commoncanvas.yaml`)

**Alternative**
The commoncanvas model is stored now in voyager at
```bash
/voyager/ceph/users/javierhn/datasets/huggingface/CommonCanvas-XL-C
``` 
You can copy it to your own ceph folder.

## Inference with optimum-habana

`commoncanvas-xl.py` is a basic script to run inference with Huggingface Diffusers and optimum-habana libraries:

```bash
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionXLPipeline

model_name='/dataset/CommonCanvas-XL-C'

scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

outputs = pipeline(
    ["A panda eating a taco"],
    num_images_per_prompt=8,
    batch_size=4,
)
image_save_dir="."
for i, image in enumerate(outputs.images):
    image.save( f"image_{i+1}.png")
```

The script loads the optimum habana libraries: scheduler and pipeline. The Diffusion pipeline is a very high level library that allows to wrap a model and run inference in a few lines of code. The scheduler is the algorithm that controls how noise is added/substracted. These have been optimizaed for the HPUs.

The script runs the pipeline with a prompt and returns 8 images that are saved on the working directory. 

## Single card run on Voyager

A pod can be launched on voyager using `cc-xl_1card.yaml` as an example. This pod runs in a single card and executes the `commoncanvas-xl.py` script. Before its execution, the `requirements.txt' installs the `optimum-habana` package.

In this `yaml` file you need to specify the location of the model (the `\dataset` volume) and the location of the working directory `\workdir` where the `commoncanvas-xl.py` is stored (and the output will be written).
