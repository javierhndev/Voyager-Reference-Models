# LLaMA for PyTorch
Here we provide the scripts and instructions to train LLaMa 2 7B model using Megatron-DeepSpeed. The code has been ported and to [Habana's HPUs](https://github.com/HabanaAI/Model-References/tree/1.11.0/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed)

## Model overview

This implementation is based on [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). Megatron is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. LLaMA training is based on [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971).

## Dataset

We are using [Redpajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) to train the LLaMA model. You can find it at
```bash
/voyager/ceph/users/czhao/datasets/llama/redpajama_tokenized
```

## Setup
A patch needs to be applied to Habana's Model-References [repository](https://github.com/HabanaAI/Model-References/tree/1.11.0/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed). In particular, a few lines of code need to be added to `/megatron/training.py`. For convenience, the fixed `training.py` is located in this folder and the yaml will automatically copy it (but needs to be placed on `RUN_PATH`) before execution. The following lines have been added:
```bash
...
  unwrapped_model = unwrap_model(model,
                                (torchDDP, LocalDDP, Float16Module))
+  if args.use_hpu:
+      unwrapped_model = [m.to('hpu') for m in unwrapped_model]
+  if mpu.get_pipeline_model_parallel_world_size() == 1:
+      args.eval_micro_batch_size = args.micro_batch_size

  if args.inference:
      optimizer = None
      lr_scheduler = None
...
```

## Training
We provide in this folder the files needed to train the model. The `llama7b_16cards.yaml` is the one who will launch an MPIJob to run the model (in 16 cards in this case). Along the yaml file you should place the `setup.sh`, `run_llama.sh` and `training.sh`.

**Note**: In the `llama7b_16cards.yaml`, you need to specify the `RUN_PATH` which is the folder where you store these four files. Also, the `results` env variable may be changed to your desired folder.

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
The image loaded in the container is based on Ubuntu 20.04. Some issues were observed if Ubuntu 22.04 was used.



