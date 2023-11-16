# Stable Diffusion for PyTorch

## Overview
Here we provide the instructions to run the stable diffusion model on Voyager. This model can be found in [Habana's Model-Referenes repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/generative_models/stable-diffusion) and it is based on latent text-to-image diffusion model.

Our example runs in a single node with 8 HPUs. Note that, using PyTorch, `hostIPC` needs to be set to `true` because the shared memory (see [PyTorch MNIST example](https://docs.habana.ai/en/latest/AWS_User_Guides/Getting_Started_Guide_EKS/mnist_example.html))

## DATASET

### Initial checkpoint
An initial checkpoint with the `first_stage_config` can be found in [https://ommer-lab.com/files/latent-diffusion/](https://ommer-lab.com/files/latent-diffusion/). Get the kl-f8.zip using `wget`, unzip it and save it somewhere in you Ceph folder. Remember to use the full path in `ckpt_path` when runing the model.

### Laion-2B-en dataset
The training uses the [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) dataset.Habana Labs provides a method to download the dataset but since it is already in Voyager you can copy directly the 16k+ files and 700Gb+. The dataset is located right now in
```bash
/voyager/ceph/users/javierhn/datasets/stable_diff/laion2B-en
```

## TRAINING

**Run on 8 HPUs**

You can use the `stable_diffusion_8cards.yaml` file in this folder to run the model. Remember to change the username to your own. The script assumes you have cloned the [Habana's Model-Referenes repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/generative_models/stable-diffusion)
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
 in your home folder in
```bash
/home/javierhn/models/Model-References
```
The `yaml` file uses this folder as a working directory and needs to be 'clean'. The output (a checkpoint file) will be written in
```bash
Model-References/PyTorch/generative_models/stable-diffusion/logs/
``` 

The `stable_diffusion_8cards.yaml` file will execute the following commands:
```bash
hl-smi;
declare -xr HOME='/scratch/tmp';
declare -xr NUM_NODES=1;
declare -xr NGPU_PER_NODE=8;
declare -xr N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

declare -xr MODEL_PATH=/home/models/Model-References/PyTorch/generative_models/stable-diffusion;
declare -xr PYTHONPATH=$PYTHONPATH:/home/models/Model-References:/home/models/Model-References/Pytorch/generative_models/stable-diffusion/src/taming-transformers:/usr/lib/habanalabs;

cd $MODEL_PATH;

mpirun  --npernode 1 \
  --tag-output \
  --allow-run-as-root \
  --prefix $MPI_ROOT \
  pip install -r requirements.txt;

declare -xr CMD="python3 $MODEL_PATH/main.py \
                 --base hpu_config_web_dataset.yaml \
                 --train \
                 --scale_lr False \
                 --seed 0 \
                 --hpus 8 \
                 --batch_size 8 \
                 --use_lazy_mode True \
                 --autocast \
                 --no-test True \
                 --max_epochs 10 \
                 --limit_train_batches 1000 \
                 --limit_val_batches 0 \
                 --val_check_interval 1000 \
                 --hpu_graph False \
                 --ckpt_path=${check_pt}\
                 --dataset_path=${dataset}";

mpirun -np ${N_CARDS} \
  --allow-run-as-root \
  --bind-to core \
  --map-by ppr:4:socket:PE=6 \
  -rank-by core --report-bindings \
  --tag-output \
  --merge-stderr-to-stdout --prefix $MPI_ROOT \
  -x PYTHONPATH \
  -x HOME \
  $CMD;

```

As a reference, an output log can be found in the folder as `output_8cards.txt`.

## Inference

### Pre-trained checkpoint
For an inference run we can use the pre-trained weights (5.7GB) from [https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/](https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/) and save the `model.ckpt` in Ceph.*Note: There were some issues to run the checkpoint from our own runs so better to use this checkpoint*.

### Run

Use `stable_diff_inference.yaml` in this directory to run inference on Voyager. The model assumes that the checkpoint file is located in
```bash
/voyager/ceph/users/javierhn/datasets/stable_diff/check_point_trained_habana
```
and the output figures will be copied to
```bash
/voyager/ceph/users/javierhn/results/stable_diffusion/inference_from_trained_habana
``` 
Remember to change it to your own folders
