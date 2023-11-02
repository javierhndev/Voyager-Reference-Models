# Hello World! Example
Here we are going to show how to run a simple 'Hello World!' on Voyager. In addition, Kubernetes commands and pod parameters are discussed.

# Running the pod

Voyager uses Kubernetes to run your applications. There are many tutorials on Kubernetes elsewhere so here we will show the basic commands.

You can find in this folder a file called `helloworld.yaml` To submit the job to the Kubernetes cluster, you need to type

```bash
kubectl create -f helloworld.yaml
```

To check the status of the pod:

```bash
kubectl get pods
```

You can check the output with

```bash
kubectl logs helloworld
```

If the run was completed your should be able to see `Hello World!` message. Finally, even if the run was completed, you need to delete the pod.

```bash
kubectl delete pod helloworld
```
