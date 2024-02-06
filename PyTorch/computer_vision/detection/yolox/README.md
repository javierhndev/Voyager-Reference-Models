# YOLOX for Pytorch
Here we provide the yaml files and instructions to train YOLOX model on Voyager.

## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.11.0/PyTorch/computer_vision/detection/yolox). This tutorial uses SynapseAI v1.11.

YOLOX is an anchor-free object detector that adopts the architecture of YOLO with DarkNet53 backbone. [Habana's repository](https://github.com/HabanaAI/Model-References/tree/1.11.0/PyTorch/computer_vision/detection/yolox) is an implementation of PyTorch version YOLOX, based on the source code from [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/MegEngine/YOLOX). More details can be found in the paper [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) by Zhen Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun.


## DATASET

The model uses the COCO 2017 dataset from [http://cocodataset.org/](http://cocodataset.org/). You can use the `data_download.yaml` to get the data which mainly will execute the following commands:
```bash
curl -O http://images.cocodataset.org/zips/train2017.zip;
unzip train2017.zip;
curl -O http://images.cocodataset.org/zips/val2017.zip;
unzip val2017.zip;
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip;
unzip annotations_trainval2017.zip;
```

 Note that, in the yaml file, you should modify the `dataset` folder to your desired folder. And remember that, during training, yo need to do it again. (so it will pass the COCO dataset location with the `--data_dir` argument).


**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/COCO
```

Feel free to use it!

## TRAINING

Single Gaudi cards have not enough memory to run the model so it needs to be run on multiple cards.

### Multi-Card Training

**Run training on 8 HPUs:**

To run on multiple cards we launch an MPI job with three files: `yolox_8cards.yaml`, `setup.sh` and `run_yolox.sh`. Some variables in the `yolox_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of your dataset.
- `output`: Output files will be written on this folder.  

Remember to change them in the launcher **and** worker.

Neither `setup.sh` or `run_yolox.sh` need to be modified. Once the MPI job is launched, the `run_yolox.sh` will execute the following:
```bash
python $MODEL_PATH/tools/train.py \
           --name yolox-s \
           --devices $N_CARDS \
           --batch-size 128 \
           --data_dir /dataset \
           --hpu max_epoch 2 output_dir /output
```
The model runs on Lazy mode with FP32 data type and train for 2 epochs.

**Run training on 16 HPUs (or more):**

To run on multiple nodes you can use the `yolox_16cards.yaml` job file. In this example we run on two nodes (16 cards) and the only difference from the `yolox_8cards.yaml` is on `NUM_NODES=2` and `replicas=2` (in the workers).



