# SSD for Pytorch
Here we provide the yaml files and instructions to train SSD model on Voyager.

## Overview

The model is supported by Intel-Habana. More details can be found in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/detection/mlcommons/SSD/ssd). This tutorial uses SynapseAI v1.13.


Single Shot MultiBox Detector (SSD) is an object detection network. For an input image, the network outputs a set of bounding boxes around the detected objects, along with their classes.

SSD is a one-stage detector, both localization and classification are done in a single pass of the network. This allows for a faster inference than region proposal network (RPN) based networks, making it more suited for real time applications like automotive and low power devices like mobile phones. This is also sometimes referred to as being a "single shot" detector for inference.

In [Habana's repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/computer_vision/detection/mlcommons/SSD/ssd) you can find their implementation of the model. which are based on [https://github.com/mlcommons/training/tree/master/single_stage_detector](https://github.com/mlcommons/training/tree/master/single_stage_detector) with certain changes for modeling and training script. Please refer to later sections on training script and model modifications for a summary of modifications to the original files.


## DATASET

The model uses the COCO 2017 dataset from [http://cocodataset.org/](http://cocodataset.org/). You can use the `data_download.yaml` to get the data which basically will execute the following commands:
```bash
curl -O http://images.cocodataset.org/zips/train2017.zip;
unzip train2017.zip;
curl -O http://images.cocodataset.org/zips/val2017.zip;
unzip val2017.zip;
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip;
unzip annotations_trainval2017.zip;
```

 Note that, in the yaml file, you should modify the `dataset` folder to your desired folder. And remember that, during training, yo need to do it again.


**Alternative**
Right now the dataset is located in
```bash
/voyager/ceph/users/javierhn/datasets/COCO
```

Feel free to use it!

### Optional: Pre-trained weights
The ResNet-34 backbone is initialized with weights from PyTorch hub file [https://download.pytorch.org/models/resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth)

By default, the code will automatically download the weights but you are free to do it yourself and save it for a later use. 

## TRAINING

Single Gaudi cards have not enough memory to run the model so it needs to be run on multiple cards.

### Multi-Card Training

**Run training on 8 HPUs:**

To run on multiple cards we launch an MPI job with three files: `ssd_8cards.yaml`, `setup.sh` and `run_ssd.sh`. Some variables in the `ssd_8cards.yaml` need to be modified to your own configuration:
- `mydir`: The location where you keep those three files.
- `dataset`: The location of your dataset.
- (Optional) `pretrained`: If you are loading the pretrained backbone manually.  

Remember to change them in the launcher **and** worker.

Neither `setup.sh` or `run_ssd.sh` need to be modified. Once the MPI job is launched, the `run_ssd.sh` will execute the following:
```bash
python $MODEL_PATH/ssd/train.py \
        -d /dataset \
        --batch-size 128 \
        --log-interval 100 \
        --val-interval 10 \
        --use-hpu \
        --hpu-lazy-mode \
        --autocast \
        --warmup 2.619685 \
        --num-workers 12
```
The model runs on Lazy mode with BF16 mixed precision, batch size 128, 12 data loader workers and trains for 10 epochs.

## Known Issues

We have noticed that after a few epochs *NaN* values start to appear.
