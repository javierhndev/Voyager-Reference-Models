# Inference of BLOOM using PyTorch
This directory provides the yaml files and instructions to run inference on different BLOOM models. The models were developed and trained by Hugginface. They were ported to [Habana's HPU accelerator](https://github.com/HabanaAI/Model-References/tree/1.11.0/PyTorch/nlp/bloom).

## Model overview

BLOOM is an autoregressive large language model. This repository is based on [Huggingface's Bigscience BLOOM model](https://bigscience.huggingface.co/blog/bloom). BLOOM comes in various configurations, varying in the size of its hidden state and number of layers, and consequently the number of parameters. Habana supports all BLOOM models up to the largest 176B using DeepSpeed for distribution across multiple Gaudi cards. **Note**: BLOOM 176B in bfloat16 requires 8 Gaudi2 cards so it is not possible to run on Voyager.



## Dataset
Before running inference you will need the necessary checkpoints. We provide the yaml files to dowload the checkpoints for 3B and 7B1 models (data_gen_bloom-3b.yaml and data_gen_bloom-7b1.yaml). The yaml files will execute the following:
```bash
export HOME=/scratch/tmp;
export PYTHONPATH=/scratch/Model-References:$PYTHONPATH;
mkdir -p /scratch/tmp/;
cd /scratch;
git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;
cd Model-References/PyTorch/nlp/bloom;
python3 -m pip install -r requirements.txt;
python3 utils/fetch_weights.py --model bigscience/bloom-7b1 --weights ${dataset};
``` 
where the `${dataset}` needs to be redefined inside the yaml and will be the folder to keep those checkpoints. Note that the `--model` argument specifies the model we want.

## INFERENCE

### Single card

In the `1card` folder you can find the `bloom-3b_1card.yaml` which can be used to run inference on the BLOMM 3B model using a single HPU. In this example, the yaml file will execute:
```bash
export HOME=/scratch/tmp;
export PYTHONPATH=/scratch/Model-References:$PYTHONPATH;
mkdir -p /scratch/tmp/;
cd /scratch;
git clone -b 1.11.0 https://github.com/HabanaAI/Model-References;
cd Model-References/PyTorch/nlp/bloom;
python3 -m pip install -r requirements.txt;
python3 ./bloom.py --weights ${dataset} --model bloom-3b --options "max_length=32" "Do you like Christmas?";
``` 
The last command executes the main script. `--model` determines the model we want to use, `max_length` is the total length of the reply (including the prompt) and the last argument is the prompt itself. Remember to modify the `dataset` variable to the location of your checkpoints.

### Multi-card (8 HPUs)
To run in multiple cards we need an MPIJob. In the `8cards` folder you can find the `bloom_8cards_mpi.yaml` file, `run_bloom.sh` and `setup.sh`.

In the `yaml` file you need to modify:
- The location of your `dataset`.
- The `RUN_PATH` folder which is where you have the `run_bloom.sh` and `setup.sh` scripts. 

Note that the model needs `DeepSpeed` which is installed when the `setup.sh` is exectued.

Finally, the `run_bloom.sh` may be modified:
- `MODEL_BLOOM` is the model you want to use. (We tried 3B and 7B1).
- The prompt is at the end of `CMD` variable. 

To run the model:
```bash
kubectl create -f bloom_8cards_mpi.yaml
```
Note that the MPIJob will be created in the *default* namespace.

