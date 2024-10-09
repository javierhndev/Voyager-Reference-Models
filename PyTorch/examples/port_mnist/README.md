# Porting your Pytorch application to Voyager
In this section we discuss how to port a Pytorch application (that has been running on GPUs) to Voyager and its Intel Gaudi (HPU) cards.

## Overview
The code in this section runs a Pytoch application on a single Gaudi card. It runs a Wide Resnet model to classify the different classes from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This Resnet model just serves as an example and could actually be substituted by many other CNN architectures. Parts of this Python script is based on the ones from the [Nvidia Deep Learning Institute](https://www.nvidia.com/en-us/training/).

Along the python script there is a yaml file which can be used to launch the application in a single card in Voyager.

## Porting the application
Here we provide the Python code `mnist_onecard.py` that will run this model. The code was originally designed for Nvidia GPUs but has been ported to Intel Gaudi HPUs including a few extra lines. For more details on HPU porting, check [Intel Gaudi documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/index.html).

The main structure of the code is:
- Import libraries
- Parse arguments
- Define model architecture
- Train function
- Test function
- "Main" function

The first steps into porting to HPU is ton include the Intel HPU libraries
```python
import habana_frameworks.torch.core as htcore
```

Then, in main, redefine the `device` for the HPU.
```python
device = torch.device('hpu')
```
In training and testing `htcore.mark_step()` must be added anytime after `loss.backward()` and `optimizer.step()`. So in our case, during training
```python
def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        labels = labels.to(device)
        images = images.to(device)

        # Forward pass 
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Setting all parameter gradients to zero to avoid gradient accumulation
        optimizer.zero_grad()

        # Backward pass
        loss.backward()
        htcore.mark_step() #<---- ADDED HERE

        # Updating model parameters
        optimizer.step()
        htcore.mark_step() #<----- ADDED HERE
```

and for the test/validation function:
```python
def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # Forward pass 
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            htcore.mark_step()  #<------- ADDED HERE

            # Extracting predicted label, and computing validation loss and validation accuracy
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    return v_accuracy, v_loss
```

## Running the application
Now that the python code is ready you can use the `mnist_1card.yaml` to launch a pod to run it. You are going to need to redefine the `/mydir` folder to the folder where you have the application. As usual, you can launch it with
```bash
kubectl create -f mnist_1card.yaml
```
Note that a `/data` folder will be created in the same location and the Fashion MNIST dataset will be downloaded (<100MB) if not already there.
