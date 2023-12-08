# UNet-2D model using TensorFlow

Here we provide the scripts and instructions to download the dataset and train the ResNet50 Keras model on Voyager using TensorFlow.  The model is mantained by Habanai Labs and you can find it in their [repository](https://github.com/HabanaAI/Model-References/tree/1.11.0/TensorFlow/computer_vision/Resnets/resnet_keras). Please check their repository for a deeper explanation of the model. The ResNet Keras model is a modified version of the original model located in [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet). It uses a custom training loop, supports 50 layers and can work with both SGD and LARS optimizers.

The model was verified on Voyager with SynapseAI version 1.11.

## Data downloading and preparation

This model uses the same dataset used for the [ResNet in Pytorch](/PyTorch/computer_vision/classification/torchvision): the ILSVRC 2012 challenge. The text below explains how to get the data if you (a) have the Pytorch dataset already or (b) you don't have it. A (c) option shows the current location of the dataset on Voyager.

### (a) Pytorch dataset available
The following steps assume you already got the dataset from [ResNet in Pytorch](/PyTorch/computer_vision/classification/torchvision). So in the folder `datasets/imagenet/ILSVRC2012` you should have:
- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar 
- /train
- /val

You should then create a new folder: `validation`. And then you can use the `data_gen.yaml` to launch a pod that will exectute the following commands:
```bash
cd ${dataset};
tar -xf ILSVRC2012_img_val.tar -C $dataset/validation;

wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt;

cd /scratch/Model-References/TensorFlow/computer_vision/Resnets;
python3 preprocess_imagenet.py --raw_data_dir=${dataset} --local_scratch_dir=${dataset}/tf_records;
```
where the `dataset`, in this example, is the `datasets/imagenet/ILSVRC2012` folder. Modify it to your own folder.

The script will generate a `./tf_records` folder which later is going to be used by the model. 

Note that it takes ~12hours to process all the data.


### (b) Download and process all the dataset
  
If you never got the dataset for the [ResNet in Pytorch](/PyTorch/computer_vision/classification/torchvision). You need to follow these steps:
- Sign up with [http://image-net.org/download-images](http://image-net.org/download-images) and acquire the rights to download original images.
- Follow the link to the 2012 ILSVRC to download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
- Use the below commands to prepare the dataset in `datasets/imagenet/ILSVRC2012`. This will create a `./train` and `./val` folders whith the original JPEG files.
  ```bash
  export dataset=datasets/imagenet/ILSVRC2012 #modify it to to your own folder
  mkdir -p $dataset/validation
  mkdir -p $dataset/train
  tar xf ILSVRC2012_img_val.tar -C $dataset/validation
  tar xf ILSVRC2012_img_train.tar -C $dataset/train
  cd $IMAGENET_HOME/train
  for f in *.tar; do
    d=`basename $f .tar`
    mkdir $d
    tar xf $f -C $d
  done
  ```  
- Then you can use the `data_gen.yaml` to launch a pod that will exectute the following commands (remember to modify `$dataset` to your own folder:
  ```bash
  cd ${dataset};
  tar -xf ILSVRC2012_img_val.tar -C $dataset/validation;

  wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt;

  cd /scratch/Model-References/TensorFlow/computer_vision/Resnets;
  python3 preprocess_imagenet.py --raw_data_dir=${dataset} --local_scratch_dir=${dataset}/tf_records;
  ```
  The script will generate a `./tf_records` folder which later it is going to be used by the model.

- Optional (but recommmended), you can execute the following commands to prepare the data for the Pytorch model. 
  ```bash
  mv $IMAGENET_HOME/validation $IMAGENET_HOME/val
  cd $IMAGENET_HOME/val
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
Note that the `yaml` file takes ~12hours to process all the data.


### (c) Directly copy the dataset from Voyager Ceph

The dataset is located now in:
```bash
/voyager/ceph/users/javierhn/datasets/imagenet/ILSVRC2012
```

## Training
### Single Card

In the `1card` folder you can find the `resnet_keras.yaml` and `resnete_keras_lars.yaml` that you can use to launch a pod that runs the ResNet model on a single HPU with or without the LARS optimizer. Note that the yaml file defines two environment variables: `dataset` and `output` (where the output will be written). You should modify the path to your own dataset and output folders.

- `resnet_keras.yaml` example: 1 HPU, batch 256, 90 epochs, BF16 precision, SGD will execute:
  ```bash
  python3 resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 -te 90 -ebe 90 -bs 256 --data_dir ${dataset} --model_dir ${output} --enable_tensorboard
  ```

- `resnet_keras_lars.yaml` example: 1 HPU, batch 256, 40 epochs, BF16 precision, LARS. It will execute:
  ```bash
  python3 resnet_ctl_imagenet_main.py -bs 256 -te 40 -ebe 40 -dt bf16 --data_dir ${dataset} --model_dir ${output} --optimizer LARS --base_learning_rate 2.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001  --single_l2_loss_op --enable_tensorboard;
  ```


### 8 Cards (single node)

To run on 8 HPUs you can use the `resnet_keras_8cards.yaml` located in th `8cards` folder. It will launch an MPIJob that will run in a single node (8 HPUs) using Horovod. The `yaml` file requires the `setup.sh` which should be saved next to it. Then, in the `yaml` file you should modify the `RUN_PATH` variable to match the location of the `setup.sh`. Note that, as for the single HPU, you also need to specify the right folders for `dataset` and `out` files.

- In this example: 8 HPUs on 1 server, batch 256, 40 epochs, BF16 precision, LARS:   
  ```bash
  python3 $MODEL_PATH/resnet_ctl_imagenet_main.py \
         --dtype bf16 \
         --data_loader_image_type bf16 \
         --use_horovod \
         -te 40 \
         -ebe 40 \
         -bs 256 \
         --optimizer LARS \
         --base_learning_rate 9.5 \
         --warmup_epochs 3 \
         --lr_schedule polynomial \
         --label_smoothing 0.1 \
         --weight_decay 0.0001 \
         --single_l2_loss_op \
         --data_dir ${dataset} \
         --model_dir ${output} \
         --enable_tensorboard
  ```
### 16 cards (multi-node)

To run on 16 HPUs (two nodes) you can use the `resnet_keras_16cards.yaml` located in th `16cards` folder. It has very few modifications compared to the `8cards`. In summary
```bash
 declare -xr NUM_NODES=2;
```
and in the Workers
```bash
replicas: 2
```



## Model Options
To see the full list of the available options and their descriptions, use the `-helpfull` command-line option, for example:
```bash
python3 resnet_ctl_imagenet_main.py --helpfull
```
