# UNet2D and UNet3D for Pytorch Lightning
Here we provide the yaml files and instructions to train the UNet2D and UNet3D models, using PyTorch Lightning, on Voyager.


## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/segmentation/Unet). This tutorial uses SynapseAI v1.15.1.



## DATASET

### Download BraTS Dataset

The dataset is stored on AWS S3. To download it, you should create a virtual environment with micromamba (or similar) and install `aws` with pip:
```bash
python3 -m pip install awscli
```
Then you can execute:
```bash
aws s3 cp s3://msd-for-monai-eu/Task01_BrainTumour.tar --no-sign-request .
``` 

### Prepare the dataset

You can use `data_gen.yaml` located in this folder to process the dataset. Remeber to specify the `dataset` variable. The yaml file will preprocess teh dataset for both UNet2D and UNet3D:
```bash
python3 preprocess.py --task 01 --dim 2 --data ${dataset} --results ${dataset}/pytorch/unet/;
python3 preprocess.py --task 01 --dim 2 --exec_mode val --data ${dataset} --results ${dataset}/pytorch/unet/;
python3 preprocess.py --task 01 --dim 2 --exec_mode test --data ${dataset} --results ${dataset}/pytorch/unet/;

python3 preprocess.py --task 01 --dim 3 --data ${dataset} --results ${dataset}/pytorch/unet/;
python3 preprocess.py --task 01 --dim 3 --exec_mode val --data ${dataset} --results ${dataset}/pytorch/unet/;
python3 preprocess.py --task 01 --dim 3 --exec_mode test --data ${dataset} --results ${dataset}/pytorch/unet/
```
If you use this configuration, your dataset for training will be located at `/dataset/pytorch/unet`.



**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/BraTS/pytorch/unet
```


## TRAINING

### Single-card
A single Gaudi card is enough to test the model. Remember to modify the `dataset` and `output` folders to the location of your Brats dataset and *output* folder.


**UNet2D**
You can find in this folder the `unet2d_1card.yaml` file to run the model.

This example will execute the model with UNet2D in lazy mode, BF16 mixed precision, batch size 64, fold 0:
```bash
python3 -u  main.py --results /output --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /dataset --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw  --exec_mode train --learning_rate 0.001 --autocast --deep_supervision --batch_size 64 --val_batch_size 64
```
**UNet3D**
To run the UNet3D on a single card you can use the `unet3d_1card.yaml`.

It will run the UNet3D in lazy mode, BF16 mixed precision, batch size 2, fold 0:
```bash
python3 -u  main.py --results /output --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /dataset --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 3 --optimizer fusedadamw  --exec_mode train --learning_rate 0.001 --autocast --deep_supervision --batch_size 2 --val_batch_size 2
```


### Multi-Card Training

**Run training on 8 HPUs:**

The execution of this model on multiple cards is slighyly different than other Pytorch models. Here we execute the mpirun `-npernode 1` and then Pythorch Lightning handles the parallelization itself.

**UNet2D**
To run on multiple cards we launch an MPI job with three files: `unet2d_8cards.yaml`, `setup.sh` and `run_unet2d.sh`. Some variables in the `unet2d_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of the Brats 2D dataset.
- `output`: Where you want to store the results.


Neither `setup.sh` or `run_unet2d.sh` need to be modified. Once the MPI job is launched, the `run_unet2d.sh` will run UNet2D in lazy mode, BF16 mixed precision, batch size 64, world-size 8, fold 0:
```bash
python3 -u $MODEL_PATH/main.py \
                      --results /output \
                      --task 1 \
                      --logname res_log \
                      --fold 0 \
                      --hpus $N_CARDS \
                      --gpus 0 \
                      --data /dataset \
                      --seed 123 \
                      --num_workers 8 \
                      --affinity disabled \
                      --norm instance \
                      --dim 2 \
                      --optimizer fusedadamw  \
                      --exec_mode train \
                      --learning_rate 0.001 \
                      --autocast \
                      --deep_supervision \
                      --batch_size 64 \
                      --val_batch_size 64 \
                      --min_epochs 30 \
                      --max_epochs 100 \
                      --train_batches 0 \
                      --test_batches 0
```

**UNet3D**
The procedure is very similar to the UNet2D.
There are three files: `unet3d_8cards.yaml`, `setup.sh` and `run_unet3d.sh`. Some variables in the `unet3d_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of the Brats 3D dataset.
- `output`: Where you want to store the results.

It will run UNet3D in Lazy mode, bf16 mixed precision, Batch Size 2, world-size 8:
```bash
python3 -u $MODEL_PATH/main.py \
                      --results /output \
                      --task 1 \
                      --logname res_log \
                      --fold 0 \
                      --hpus $N_CARDS \
                      --gpus 0 \
                      --data /dataset \
                      --seed 1 \
                      --num_workers 8 \
                      --affinity disabled \
                      --norm instance \
                      --dim 3 \
                      --optimizer fusedadamw  \
                      --exec_mode train \
                      --learning_rate 0.001 \
                      --autocast \
                      --deep_supervision \
                      --batch_size 2 \
                      --val_batch_size 2 \
                      --min_epochs 6 \
                      --max_epochs 20 
```



