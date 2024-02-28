# LLaMA for PyTorch
Here we provide the scripts and instructions to train LLaMa 7B model using Megatron-DeepSpeed. The code has been ported to [Habana's HPUs](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed)

## Model overview

This implementation is based on [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). Megatron is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. LLaMA training is based on [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971).

## Dataset

We are using [Redpajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) to train the LLaMA model. You can find it at:
```bash
/voyager/ceph/users/czhao/datasets/llama/redpajama_tokenized
```


## Training
We provide in this folder the files needed to train the model. The `llama7b_16cards.yaml` will launch an MPIJob to run the model (on 16 cards in this case). Along the yaml file you should place the `setup.sh` and `run_llama.sh`.

**Note**: In the `llama7b_16cards.yaml`, you need to specify the `my_path` which is the folder where you store these three files. Also, the  `output`  variable should be changed to your desired folder. Change them in BOTH launcher AND worker.

To run the LLaMA model:
```bash
kubectl create -f llama7b_16cards.yaml
```
which will be created on the default namespace.

This will first install the necessary packages (including DeepSpeed) using `setup.sh` and then launch the model with `run_llama.sh`.

`run_llama.sh` may be modified to tune some parameters used in the LLaMA model.

### Scale out
The provided `yaml` file already usses 2 nodes but if, for example, you need to run in 8 nodes (64) you should only modify `NUM_NODES=8` and `replicas: 8`.


## Known Issues
The image loaded in the container is based on Ubuntu 20.04. Some issues were observed when Ubuntu 22.04 was used.



