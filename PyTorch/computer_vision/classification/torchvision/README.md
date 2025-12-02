# Computer vision classification with Pytorch
Here we provide the yaml files and instructions to train ResNet50 and ResNeXt101 models on Voyager.

## Overview

The models are supported by Intel Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision). This tutorial uses SynapseAI v1.21.4. The base model used here is actually from [GitHub: PyTorch-Vision](https://github.com/pytorch/vision/tree/release/0.10/torchvision/models) which has been modifed by Habana. The following models have been tested on Voyager:

- ResNet50
- ResNext101

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

**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012
```

Feel free to use it!

## TRAINING

We are showing here some examples for each model. You can find more examples with differen parameters in [Habana's repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision)
- To see the available training parameters for ResNet50 and ResNeXt101, run:
  ```bash
  python3 -u train.py --help
  ```

All the yaml files here use two environment variables: `dataset` and `output`. Use them to point the location of the dataset and output folder (Use Ceph). Multi-card examples also define a path for a `RUN_PATH` (where the `.yaml` file and `setup.sh` are located).
 
Note that the number of epochs for every run has been set to 1 for testing.

### Training examples (single card and multi-card)
**Run on 1 HPU**
You can find the yaml files in the `1card` folder. Execute them with
```bash
kubectl create -f themodel.yaml
```
to launch a pod to run the model. The code is dowloaded in to the scratch. The location of the dataset of the model is defined in `dataset` variable. You can also choose your output folder with the `output` environment variable. 

Note each run takes ~3 hours to run 1 epoch in 1 HPU.

- ResNet50 (lazy mode, BF16 mixed precision, batch Size 256, custom learning rate, Habana dataloader, Eager mode, torch compile):
  ```bash
  kubectl create -f resnet50_1card.yaml 
  ```
  which will execute the following:
  ```bash
  python3 -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /dataset --output-dir /output --save-checkpoint --epochs 1 --autocast  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80 --run-lazy-mode=False --use_torch_compile 
  ```
- ResNeXt101 (lazy mode, BF16 mixed precision, batch size 256, custom learning rate, Habana dataloader, Eager mode, torch compile):
  ```bash
  kubectl create -f resnext101_1card.yaml 
  ```

**Run on 8 HPUs**
To run the models in multiple cards on Voyager, we can use `mpirun` in a single pod. 

The following commands are executed in this pod:

```bash
export PYTHONPATH=/scratch/Model-References:$PYTHONPATH;
mkdir -p /scratch/tmp/;
cd /scratch;
export N_CARDS=8;
git clone -b 1.21.0 https://github.com/HabanaAI/Model-References;
cd Model-References/PyTorch/computer_vision/classification/torchvision;
pip install -r requirements.txt;
export CMD="python3 train.py \
              #model parameters";

mpirun -n ${N_CARDS} \
       --allow-run-as-root \
       --bind-to core \
       --map-by ppr:4:socket:PE=6 \
       -rank-by core --report-bindings \
       --tag-output \
       --merge-stderr-to-stdout \
       -x PYTHONPATH \
       $CMD;
```


- ResNet50 (lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs, Eager mode, torch compile): 
  ```bash
  kubectl create -f resnet50_8cards.yaml
  ``` 
- ResNeXt101 (lazy mode, BF16 mixed precision, batch size 256, 8 HPUs, uses habana_dataloader, Eager mode, torch compile)
  ```bash
  kubectl create -f resnext101_8cards.yaml
  ```

**Run on 16 or more HPUs**

To run the models in multiple cards on Voyager, we submit an MPIJob instead of a single pod. We are providing as an example the ResNext101 model on 16. To run in more than two nodes change the values in `NUM_NODES` and `Replicas` (in workers) in the yaml file.

The following commands are executed in the MPIJob:

```bash
declare -xr HOME='/scratch/tmp';
declare -xr NUM_NODES=2;
declare -xr NGPU_PER_NODE=8;
declare -xr N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

declare -xr RUN_PATH=/home/models/resnet/8cards;
declare -xr MODEL_PATH=/scratch/Model-References/PyTorch/computer_vision/classification/torchvision;

declare -xr PYTHONPATH=$PYTHONPATH:/scratch/Model-References;

HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
declare -xr MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
declare -xr MASTER_PORT=${MASTER_PORT:-15566};

echo $MASTER_ADDR;
echo $MASTER_PORT;

mpirun  --npernode 1 \
  --tag-output \
  --allow-run-as-root \
  --prefix $MPI_ROOT \
  -x MODEL_PATH \
  -x HOME \
  $RUN_PATH/setup.sh;

declare -xr CMD="python3 $MODEL_PATH/train.py \
                 #model parameters... "

mpirun -np ${N_CARDS} \
  --allow-run-as-root \
  --bind-to core \
  --map-by ppr:4:socket:PE=6 \
  -rank-by core --report-bindings \
  --tag-output \
  --merge-stderr-to-stdout --prefix $MPI_ROOT \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x PYTHONPATH \
  -x HOME \
  $CMD;
```


### Profiling

The models show a good scaling when number of nodes is increased. As an example, the following table shows the training time of 1 Epoch using the ResNet50 model

|         | Time    |
| ------- | ------- |
| 1 HPU   | ~3hours |
| 8 HPUs  | 20 min  |
| 16 HPUs | 11 min  |
| 32 HPUs | 7 min   |


