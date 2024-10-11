# Parallelize your Pytorch application on Voyager
In this section, we show how to parallelize one Pytorch application (which has been already ported to HPUs) using Pytorch Distributed Data Parallel (DDP) and the Habana's HCCL package as backend.

## Overview
We are showing how to parallelize the MNIST models we have been using in the [how to port your app to Voyager](/PyTorch/examples/port_mnist) section. It runs a Wide Resnet model to classify the different classes from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This simple Resnet model just serves as an example and it could actually be substituted by many other CNN architectures. Parts of this Python script is based on the ones from the [Nvidia Deep Learning Institute](https://www.nvidia.com/en-us/training/).

The parallelization is done using Pytorch's [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and the [HCCL](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html) package as backend (which is Habana's implementation of Nvidia's NCCL).

We provide also two yaml files to launch the appplication on one (8 cards) and two (16 cards) nodes.

## YAML files
You can use  `mnist_8cards.yaml` or `mnist_16cards.yaml` to run the model on one or two nodes respectively. Note that the `mydir` volume needs to be redefined to the location of the scripts.

These yamls files will launch an MPIJob which is the default way to run MPI on Voyager. Then, the following commands will be executed:
```bash
declare -xr NUM_NODES=1;
declare -xr NGPU_PER_NODE=8;
declare -xr N_CARDS=$((NUM_NODES*NGPU_PER_NODE));


HOSTSFILE=${HOSTSFILE:-$OMPI_MCA_orte_default_hostfile};
declare -xr MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p);
declare -xr MASTER_PORT=${MASTER_PORT:-15566};

echo $MASTER_ADDR;
echo $MASTER_PORT;

sleep 20s;

declare -xr CMD="python3 /mydir/mnist_multicard.py \
                   --epochs 20 \
                   --target-accuracy 0.90";

mpirun -np ${N_CARDS} \
  --allow-run-as-root \
  --bind-to core \
  --map-by ppr:4:socket:PE=6 \
  -rank-by core --report-bindings \
  --tag-output \
  --merge-stderr-to-stdout --prefix $MPI_ROOT \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  $CMD;
```
The `NUM_NODES` is the number of nodes used and must match the `Workers` in the MPIJob. It is used to calculate the total number of Gaudi cards to be used (8 per node). The `MASTER_ADDR` and `MASTER_PORT` are defined and later passed to the mpirun. No need for users to modify them.
The `CMD` contains the command we want to execute which in this case is this pytorch application. Note that here you can pass arguments to the model.

Finally, the `mpirun` command will execute the `$CMD` using MPI.

## Parallelazing the Pytorch application
We describe the example from `mnist_multicard.py` to show how to parallelize a working Pytorch application using DDP. Note that this is the same model used in [how to port your app to Voyager](/PyTorch/examples/port_mnist) section but now we run it on multiple cards. Most of the code will remain the same (model architecture, training and testing functions...) and only the `main` function needs some modifications. You can use `vimdiff` or similar to highlight the differences.

**Import packages**

First, some packages need to be added:
```python
import habana_frameworks.torch.distributed.hccl
from habana_frameworks.torch.distributed.hccl import initialize_dist
import torch.distributed as dist
```
The `torch.distributed` is Pytorch Distributed communication package to run parallel jobs. It contains the `Distributed Data-Parallel (DDP)` module that allows to distribute the data amonf different HPUs to accelrate training.
`HCCL` is the Intel Habana's package that emulates NVIDIA Collective Communication Library (NCCL). That is the backend in this parallelization.

**Initialization**

Then, in the `main` function, we need to add the following at the beginning:
```python
world_size, rank, local_rank = initialize_distributed_hpu()
print('Hi Im Worker:',rank)  

dist.init_process_group(
     backend='hccl',
     world_size=world_size,
     rank=rank
     )
```
The `initialize_distributed_hpu()` will return world_size(total number of cards), rank (worker ID in world), local_rank (worker ID in the node). Then the `dist.init_process_group` will initialize the parallel process.

**Data download**

Only one worker should download the data. (Note that now, data is downloaded in to your `/mydir` folder). So:
```python
download = True if local_rank == 0 else False
if rank == 0:
   train_set = torchvision.datasets.FashionMNIST("/mydir/data", download=download, transform=
                                               transforms.Compose([transforms.ToTensor()]))
   test_set = torchvision.datasets.FashionMNIST("/mydir/data", download=download, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  
dist.barrier() #all workers wait until download complete
                                                                      
if rank != 0:
  train_set = torchvision.datasets.FashionMNIST("/mydir/data", download=download, transform=
                                              transforms.Compose([transforms.ToTensor()]))
  test_set = torchvision.datasets.FashionMNIST("/mydir/data", download=download, train=False, transform=
                                              transforms.Compose([transforms.ToTensor()]))   
```
**Data distribution**

Before loading the data into the `Dataloader`, it needs to be distributed among the different workers using the `DistributedSampler` class. So train and test data need their own sampler as: 

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=world_size,
    rank=rank
    )

test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_set,
    num_replicas=world_size,
    rank=rank
    )

# Training data loader
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size, drop_last=True, sampler=train_sampler)
# Validation data loader
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=args.batch_size, drop_last=True, sampler=test_sampler)
```

**Create model**

Create the model in the HPUs and implement distributed data parallelism on it.
```python
model=WideResNet(num_classes).to(device)
model = nn.parallel.DistributedDataParallel(model)
```

**Gather output data from different workers**

In the main loop, some commands need to be added. First, for each epoch, the sampler needs to be shuffled. Then, a reduction operation is used to obtain the total `image_per_second` value. Finally, to track validation accuracy and loss, we need to call an `all_reduce` operation to calculate them as well.
```python
for epoch in range(args.epochs):

    t0 = time.time()

    # update the random seed of the DistributedSampler to change
    # the shuffle ordering for each epoch. It is necessary to do this for
    # the train_sampler, but irrelevant for the test_sampler. The random seed
    # can be altered with the set_epoch method (which accepts the epoch number
    # as an input) of the DistributedSampler. 
    train_sampler.set_epoch(epoch)

    train(model, optimizer, train_loader, loss_fn, device)

    # At the end of every training epoch, synchronize (using dist.barrier())
    # all processes to compute the slowest epoch time. 
    dist.barrier()
    epoch_time = time.time() - t0
    total_time += epoch_time

    images_per_sec = torch.tensor(len(train_loader)*args.batch_size/epoch_time).to(device)
    torch.distributed.reduce(images_per_sec, 0)

    v_accuracy, v_loss = test(model, test_loader, loss_fn, device)

    #average validation accuracy and loss across all GPUs
    torch.distributed.all_reduce(v_accuracy, op=dist.ReduceOp.SUM)
    v_accuracy=v_accuracy/world_size
    torch.distributed.all_reduce(v_loss, op=dist.ReduceOp.SUM)
    v_loss=v_loss/world_size
    val_accuracy.append(v_accuracy)

    if rank == 0:
        print("Epoch = {:2d}: Cumulative Time = {:5.3f}, Epoch Time = {:5.3f}, Images/sec = {}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}".format(epoch+1, total_time, epoch_time, images_per_sec, v_loss, val_accuracy[-1]))

    if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
        print('Early stopping after epoch {}'.format(epoch + 1))
        break

```
