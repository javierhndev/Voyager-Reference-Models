# MPIJobs in Voyager
This section describes how to run an application in multiple HPUs on Voyager.

As an example, we are runing the MNIST model supported by [Habana](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/hello_world).

## Overview

Voyager has 42 nodes and each node contains 8 HPUs. To run multi-process applications we can use the `MPIJob` object in Kubernetes. This will allow us to do an `mpirun` and run in one or multiple Voyager's nodes.

The application in this example is the MNIST model, developed by [Habana](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/hello_world) and based on the [TensorFlow tutorial](https://www.tensorflow.org/datasets/keras_example), which uses Horovod to run in multiple HPUs.

## SETUP

To get the MNIST model you need to clone the [Habana Model-References](https://github.com/HabanaAI/Model-References) somewhere in your home folder (you are going to need this folder in the `yaml` file).
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```


# TO DO!!!!

