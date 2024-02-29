# BERT model using TensorFlow
Here we provide the scripts and files to download the dataset and train a BERT model on Voyager. The model is mantained by Habana and you can find it in their [repository](https://github.com/HabanaAI/Model-References) in the `Natural language processing` section. Please check their repository for a deeper explanation of the BERT model.

The model was verified on Voyager with SynapseAI version 1.13.

### Main steps
- Download the dataset:
  - Download the Wikipedia and BookCorpus datasets.
  - Pack training dataset (*to do, issue with memory*) for better performance.
  - Download the fine-tuning datasets: MRPC and SQuAD.
  - Download the pretrained model and config. The config file is needed even if you want to do the pre-training.
- Training:
  - Pretrain phase 1.
  - Pretrain phase 2.
  - Fine-tuning with the MRPC dataset.
  - (*To do*)Fine-tuning with the SQuAD dataset.

## Data downloading
The required scripts to download, extract and preprocess the datasets are located in the `Model-References/TensorFlow/nlp/bert/data_preprocessing`folder in Habana's repository.

Alternatively, you can copy the dataset from:

```bash
/voyager/ceph/users/javierhn/datasets/bert
```

### Download pre-training dataset

Ideally the dataset consists on the Wikiepdia and BookCorpus datasets but by default, following Habana's suggestions, we skip the BookCorpus as it gets many errors and broken links. The required scripts to download and extract the datasets are located in the `Model-References/TensorFlow/nlp/bert/data_preprocessing`folder. But this default method in Habana's repository needs to be slighlty modified to run on Voyager.

The yaml file `data_gen.yaml` in our folder can be executed with `kubectl` to download and extract the data. It takes several hous and 100GB+ of data. Follow the next steps to succesfully run it.

Clone the [Habana Model-References repository](https://github.com/HabanaAI/Model-References) somewhere in you `/home`. Use the `hl-smi` command to determine SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Inside the repository, modify the  `Model-References/TensorFlow/nlp/bert/data_preprocessing/create_datasets_from_start.sh` file. Change
```bash
BERT_PREP_WORKING_DIR=${2:-"/data/tensorflow/bert/books_wiki_en_corpus"}  
```
for the folder in `Ceph` you want to store the data. For example
```bash
BERT_PREP_WORKING_DIR=${2:-"/voyager/ceph/users/yourusername/datasets/bert/books_wiki_en_corpus"
``` 

An additional file needs to be modified it the Habana's repository. The file is `Model-References/TensorFlow/nlp/bert/data_preprocessing/WikiDownloader.py` and the command `lbzip2` should be `bzip2` (because the package is not installed by default in Ubuntu). **Optional**: The `bzip2` unpacking may take some time (1+ hour) so better to move that piece of code to inside the *if* statement if you neeed to run it again.


Beofre launching the pod take into account the following. The `yaml` file assumes the repository is located in:
```bash
cd /home/youruser/models/Model-References
```
so modify it to your own folder. Also the `Ceph` should be mounted to your own folder.

Finally, to launch the pod:

```bash
kubectl create -f data_gen.yaml 
```

### Packing Pre-training Datasets
*TO DO*

Supported by Habana but not required to run on Voyager


### Download Fine-tuning Datasets

The MRPC dataset can be easily dowloaded using the `download_dataset.py`. Include, as a parameter, the folder you want to store the data. For example:
``` bash
python Model-References/TensorFlow/nlp/bert/download/download_dataset.py /voyager/ceph/users/youruser/datasets/bert/MRPC
```
Remeber to use this folder when runing the fine-tuning with the `run_classifier.py`.

The v1.1 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset needs to be manually downloaded (we recommend to store it in your Ceph folder). To download the data, execute the following:

```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

### Download the pretrained model
Habana offers a pretrained model that can be used for fine-tuning. Even if you are not interested in the pretrained model you need to download it because you need the `bert_config.json` to train your model.

Download the BERT Large pretrained model using our `pretrained.yaml` file. Change the *output* variable to your own folder.

## Training
### Single Card

- The pretraining of **Phase 1** of the BERT Large can be done using the `bert_phase1_1card.yaml` in the *1card* folder.Launch the pod with
  ```bash
  kubectl create -f bert_phase1_1card.yaml
  ```   
  Remeber to change the *dataset, pretrained parameters* and *output* folders to your own. The Single card run is too slow but can be used to test the model.

- **Phase 2** has not been checked because Phase 1 is too slow in a single card.

- Fine-tuning using the **MRPC** dataset:
  You can use the `bert_finet_MRPC_1card.yaml` to do the fine-tuning on the pretrained model provided by Habana. Even with a single HPU the run takes only a few minutes.
  ```bash
  kubectl create -f bert_finet_MRPC_1card.yaml 
  ```
  Remember to change the *input*,*output* and *pretrain* folders to your own.

- Fine-tuning using the **SQuAD** dataset:
  *TO DO*. 

### 8 Cards (single node)

Most of the parameters passed to the model are the same as in a single card. But the *yaml* files have been modified to run in multiple HPUs. Here we launch an MPIJob to run the model in one node (with 8 cards). Note that we have modified the number of steps to run it in a few minutes.

- Pretraining **Phase 1** of BERT Large. The MPIJob can be launched using the `bert_phase1_8cards.yaml` located in the `8 cards` folder. You need to modify the `mydir` variable to point to the location of your `yaml` and `setup.sh` files. 
 
  Remeber to change the *dataset, pretrained parameters* and *output* folder to your own. In the launcher AND worker.

  The number of steps have been reduced to `num_train_steps=20` to check the model. It takes about ~20 minutes in eight HPUs.

- Pretraining of **Phase 2**. You can use `bert_phase2_8cards.yaml` to launch the MPIJob to train the model. The `init_checkpoint` parameter takes as input the checkpoint from phase 1. Again, remeber to change the folders to your own.

  We reduced the number of steps to 10 and it took ~15 minutes to run.

- Fine-tuning with the **MRPC** dataset. Launch the MPIJob in 8 cards using the `bert_finet_MRPC_8cards.yaml` file. You should use the last checkpoint from Phase 2 fo the parameter `init_checkpoint` in the script.

  Run takes a couple of minutes.    

## Known issues
 The model can run in multiple nodes but it fails during the checkpoint save. 
