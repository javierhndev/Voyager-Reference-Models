# UNet-2D model using TensorFlow
Here we provide the scripts and instructions to download the dataset and train a UNet-2D model on Voyager using TensorFlow. The model is mantained by Habanai Labs and you can find it in their [repository](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/computer_vision/Unet2D). Please check their repository for a deeper explanation of the UNet2D model. The UNet Medical model is a modified version of the original model located in [NVIDIA UNet Medical Image Segmentation for TensorFlow 2.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical).

The model was verified on Voyager with SynapseAI version 1.13.

## Data downloading
First step is to download the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home) using the `download_dataset.py` script inside Habana's repository. However, at the time this model was tested on Voyager, the link was broken. Fortunately, the dataset is mirrored on [Kaggle](https://www.kaggle.com/soumikrakshit/isbi-challenge-dataset). The dataset is relatively small and it consists on three `tif` files. You can save them in your Ceph folder and use them when running the model using the `--data_dir` parameter.

You can also find the dataset in
```bash
/voyager/ceph/users/javierhn/datasets/UNet2D_tf
```  

## Training
### Single Card

In this folder you can find the `unet2d_tf.yaml` that you can use to launch a pod that runs the UNet-2D model on a single HPU.

- In this example: 1 HPU training with batch size 8, bfloat16 precision and fold 0.
  ```bash
  python3 unet2d.py --data_dir /dataset --batch_size 8 --dtype bf16 --model_dir /output --fold 0 --tensorboard_logging; 
  ```
Note that the yaml file defines two volumes: `dataset` and `output` (where the output will be written). You should modify the path to your own dataset and output folders.

### 8 Cards (single node)

To run on 8 HPUs you can use the `unet2d_tf_8cards.yaml` located in this folder. It will launch an MPIJob that will run in a single node (8 HPUs) using Horovod. The `yaml` file requires the `setup.sh` which should be saved next to it. Then, in the `yaml` file you should modify the `my_dir` volume to match the location of the `setup.sh`. Note that, as in the single HPU, you also need to specify the right paths for `dataset` and `output`. Remember to do if for the launcher AND worker.

- In this example: 8 HPUs training with batch size 8, bfloat16 precision and fold 0   
  ```bash
  python3 $MODEL_PATH/unet2d.py --data_dir /dataset \
          --batch_size 8 \
          --dtype bf16 \
          --model_dir /output \
          --fold 0 \
          --tensorboard_logging \
          --log_all_workers \
          --use_horovod
  ```

### 16 cards (multi-node)

The model is able to run on multiple nodes. Here, we provide an example with 2 nodes (16 HPUs) using `unet2d_tf_16cards.yaml`. Note that very little changes are needed to run on multiple nodes. Only `NUM_NODES` and `replicas` need to be modified.

## Model Options
To see the full list of the available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
python3 unet2d.py --help
```
