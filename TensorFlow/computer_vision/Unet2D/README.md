# UNet-2D model using TensorFlow
Here we provide the scripts and instructions to download the dataset and train a UNet-2D model on Voyager using TensorFlow. The model is mantained by Habanai Labs and you can find it in their [repository](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/computer_vision/Unet2D). Please check their repository for a deeper explanation of the UNet2D model. The UNet Medical model is a modified version of the original model located in [NVIDIA UNet Medical Image Segmentation for TensorFlow 2.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical).

The model was verified on Voyager with SynapseAI version 1.11.

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
  python3 unet2d.py --data_dir ${dataset} --batch_size 8 --dtype bf16 --model_dir ${out} --fold 0 --tensorboard_logging; 
  ```
Note that the yaml file defines two environment variables: `dataset` and `out` (where the output will be written). You should modify the path to your own dataset and output folders.

### 8 Cards (single node)

To run on 8 HPUs you can use the `unet2d_tf_8cards.yaml` located in this folder. It will launch an MPIJob that will run in a single node (8 HPUs) using Horovod. The `yaml` file requires the `setup.sh` which should be saved next to it. Then, in the `yaml` file you should modify the `RUN_PATH` variable to match the location of the `setup.sh`. Note that, as for the single HPU, you also need to specify the right folders for `dataset` and `out` files.

- In this example: 8 HPUs training with batch size 8, bfloat16 precision and fold 0   
  ```bash
  python3 $MODEL_PATH/unet2d.py --data_dir ${dataset} \
          --batch_size 8 \
          --dtype bf16 \
          --model_dir ${out} \
          --fold 0 \
          --tensorboard_logging \
          --log_all_workers \
          --use_horovod
  ```

## Model Options
To see the full list of the available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
python3 unet2d.py --help
```
