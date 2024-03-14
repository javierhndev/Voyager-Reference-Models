# Simple TensorFlow model
This is an example on how to run a python script (that executes a deep learning model with TensorFlow) on Voyager. The code shows how to implement a simple classification model used for computer vision. It uses the `Fashion-MNIST` dataset which consists of thousands of [Zalando's](https://github.com/zalandoresearch/fashion-mnist) article images.

The purpose of this code is to test TensorFlow with a very simple example. The model is NOT optimized to run on Habana's HPU accelerators.

You can find the following in the `test_tf.yaml` file
```bash
apiVersion: v1
kind: Pod
metadata:
  name: testtf
spec:
  restartPolicy: Never
  volumes:
   - name: workdir
     hostPath:
        path: /home/yourusername/test_tf
        type: Directory
  containers:
    - name: gaudi-container
      image: vault.habana.ai/gaudi-docker/1.13.0/ubuntu22.04/habanalabs/tensorflow-installer-tf-cpu-2.13.1
      volumeMounts:
         - mountPath: /workdir
           name: workdir
      resources:
        limits:
          memory: 32G
          cpu: 1
          habana.ai/gaudi: 1
        requests:
          memory: 32G
          cpu: 1
          habana.ai/gaudi: 1
      command: ["/bin/sh","-c"]
      args: ["python3 /workdir/example_tf_mnist.py"]
```

In this pod, you can see how to load a folder on Voyager (Remember to change the path in `workdir` to your own folder) and how to run a python script.
