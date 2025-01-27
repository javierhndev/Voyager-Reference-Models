# Running Jupyter notebooks on Voyager

In this tutorial we are showing how to run a Jupyter Notebook on Voyager and its Intel Gaudi (HPU) cards.


### 1) Launch a pod that runs Jupyter Notebook
The first step is to launch a pod that installs and executes Jupyter in one of the Gaudi cards. You can use as an example the `jupyter_pod.yaml` file located in this folder. Remember to load your working directory or any other folders you may be interested in this yaml file.

This example pod will create a container with Habana's image , install jupyter via pip and execute jupyter using the following arguments:
```bash
pip install jupyter;
jupyter notebook --allow-root;
```

To launch this pod:
```bash
kubectl create -f jupyter_pod.yaml 
```

### 2) Get the link to the Jupyter Notebook
You can check the pod status with `kubectl get pods`. After a few seconds Jupyter notebook should be installed and ready. Type:
```bash
kubectl logs thepodname
```

and at the end of the log, you should see the links you will use shortly. You should see something similar to this:

```bash
[C 2025-01-27 18:04:27.192 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-48-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=afdf0fe8322e417c4f76ee316b22c0b8e5cb6b021ab5668b
        http://127.0.0.1:8888/tree?token=afdf0fe8322e417c4f76ee316b22c0b8e5cb6b021ab5668b

```

These are the links you will need later to run Jupyter Notebook from your pc. Be aware of sharing them with anyone as they work as a password to your notebooks.

### 3) Port forward
You need to port-forward your pod. Using for example the following kubectl command in Voyager:
```bash
kubectl port-forward jupyter-notebook-pod 9888:8888
``` 
If the 9888 port is used by someone else, you can change it to a different number.

### 4) Set the ssh tunnel
Now, from you pc, you can create an ssh tunnel with the following command:
```bash
ssh -N -f -L 9888:localhost:8888 login.voyager.sdsc.edu
``` 

### 5) Copy the link to your browser

Now, get one of the URL from step 2 and copy it to your folder. Jupyter Notebook should work in your browser!

To verify everything is working. Open a notebook and run the following command:
```bash
!hl-smi
```

It should give the information of the Gaudi card you are running on.

### Last step: Delete the pod when you are done

Once you have finished your session with Jupyter Notebook, don't forget to delete the pod with
```bash
kubectl delete pod nameofpod
```

If not, the pod will keep running indefinitely, even if the Jupter Notebook is closed, and you may be charged for that extra time.

