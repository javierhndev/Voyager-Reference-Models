#(DEPRECATED)

# MPIJobs in Voyager
This section describes how to run an application in multiple HPUs on Voyager.

As an example, we are runing the MNIST model supported by [Habana](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/hello_world).

## Overview

Voyager has 42 nodes and each node contains 8 HPUs. To run multi-cards applications we can use the `MPIJob` object in Kubernetes. This will allow us to do an `mpirun` and run in one or multiple Voyager's nodes.

The application in this example is the MNIST model, ported by [Habana](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/hello_world) and based on the [TensorFlow tutorial](https://www.tensorflow.org/datasets/keras_example), which uses Horovod to run in multiple HPUs.

The example below shows how to run the application in 8 HPUs (one node). And, at the end of this README, an exmaple on 16 HPUs is discussed/

## SETUP

To run this example you will need the `test8cards_mnist_TF.yaml` and the `setup.sh` files in this folder and copy them somewhere in your home folder. The yaml file needs to know the location of the setup.sh (ideally next to it) and set that location in the `my_dir` variable. Note that `my_dir` needs to be defines in the launcher AND worker.

The yaml file will launch the MPIJob in one node (8 HPUs), define some environment variables, copy the code to `~/scratch` and run it. The following commands (with comments) are executed:
```bash
declare -xr NUM_NODES=1;
declare -xr NGPU_PER_NODE=8;
declare -xr N_CARDS=$((NUM_NODES*NGPU_PER_NODE));

declare -xr RUN_PATH=/mydir;
declare -xr SYNAPSE_AI_VER=1.13.0;
declare -xr MODEL_PATH=/scratch/Model-References/TensorFlow/examples/hello_world;
declare -xr PYTHONPATH=$PYTHONPATH:/scratch/Model-References;
sleep 15s;

#this mpirun execute the setup.sh which download the code from Habana's repository
# to scratch. In more complicated examples, setup.sh is used to install libraries or other dependecies
mpirun  --npernode 1 \
  --tag-output \
  --allow-run-as-root \
  --prefix $MPI_ROOT \
  -x MODEL_PATH \
  -x SYNAPSE_AI_VER \
  $RUN_PATH/setup.sh;

# CMD is the command (the code) you want to run in multiple cards
declare -xr CMD="python3 $MODEL_PATH/example_hvd.py";

mpirun -np ${N_CARDS} \
  --allow-run-as-root \
  --bind-to core \
  --map-by ppr:4:socket:PE=6 \
  -rank-by core --report-bindings \
  --tag-output \
  --merge-stderr-to-stdout --prefix $MPI_ROOT \
  -x PYTHONPATH \
  $CMD;
```
The `setup.sh` in this example is quite simple and only clone Habana's repository to `/scratch`. It executes the following:
```bash
#!/bin/sh

cd /scratch;

git clone -b $SYNAPSE_AI_VER https://github.com/HabanaAI/Model-References;
```

## Execution
To create the MPIJob, you can execute the command
```bash
kubectl create -f test8cards_mnist_TF.yaml 
```

At this moment MPIJobs are runing in the default namespace, so to check their status you can
```bash
kubectl get pods -n default
```

And to check their ouputs
```bash
kubectl logs launcher_id -n default
```
where the `launcher_id` is the full ID that has been assigned to the launcher pod.

This MPIJob example stays on the system once is completed so be sure to deleted with
```bash
kubectl delete -f test8cards_mnist_TF.yaml
```

## Multiple-node example
The above example can be run on 16 HPU cards (two nodes) or more. To do it, you can use the`test16cards_mnist_TF.yaml` file. 
There are just a few differences between the *8cards* ans *16cards*. `NUM_NODES` should be the number of nodes you want to use (2 in this case) as well as `replicas` (in workers). 
