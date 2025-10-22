# Hello World! Example
Here we are going to show how to run a simple 'Hello World!' on Voyager. In addition, Kubernetes commands and pod parameters are discussed.

# Running a pod

Voyager uses Kubernetes to run your applications. There are many tutorials on Kubernetes elsewhere so here we will show the basic commands.

You can find in this folder a file called `helloworld.yaml` which looks like this:
```bash
apiVersion: v1
kind: Pod
metadata:
  name: test
  namespace: yourusername
spec:
  restartPolicy: Never
  containers:
    - name: gaudi-container
      image: vault.habana.ai/gaudi-docker/1.21.4/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
      resources:
        limits:
          memory: 32G
          cpu: 12
          habana.ai/gaudi: 1
      command: ["/bin/sh","-c"]
      args:
        - echo 'Hello World!';
```
The yaml file creates a pod and loads a container with Habana's image and a single HPU (`habana.ai/gaudi: 1`).

A pod is the simplest object that runs in Kubernetes.

In this pod, you will need to specify your namespace which is assigned to you when you get an allocation in the Voyager system.

To run the pod, use the command

```bash
kubectl create -f helloworld.yaml
```

To check the status of the pod:

```bash
kubectl get pods -n yournamespace
```

You can check the output with

```bash
kubectl logs test -n javierhn-sds275
```

If the run was completed your should be able to see `Hello World!` message. Finally, even if the run was completed, you need to delete the pod.

```bash
kubectl delete pod test -n yournamespace
```
