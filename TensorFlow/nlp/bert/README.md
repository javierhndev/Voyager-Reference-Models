# BERT model using TensorFlow
Here we proivde the scripts and files to dowload the datsset and train a BERT model on Voyager. The model is mantained by Habana and you can find it in their [repository](https://github.com/HabanaAI/Model-References) in the `Natural language processing` section. Please check their repository for a deeper explanation of the BERT model.

The model was verified on Voyager with SynapseAI version 1.11.

### Main steps
- Download the dataset:
  - Download the Wikipedia and BookCorpus (*to do*) datasets.
  - Pack training dataset (*to do, issue with memory*)
  - Download the fine-tuning datasets: MRPC and SQuAD.
  - Download the pretrained model and config. The config file is needed even if you want to to the pre-training.
- Training:
  - Pretrain phase 1.
  - Pretrain phase 2.
  - Fine-tuning with the MRPC dataset.
  - Fine-tuning with the SQuAD dataset.

## Data downloading
The required scripts to download, extract and preprocess the datasets are located in the `Model-References/TensorFlow/nlp/bert/data_preprocessing`folder in Habana's repository.

Alternatively, you can copy the dataset from:

```bash
/voyager/ceph/users/javierhn/datasets/bert
```

### Download pre-training dataset

Ideally the dataset consists on the Wikiepdia and BookCorpus datasets but by default, following Habana's suggestions, we skip the BookCorpus as it gets many errors and broken links. The required scripts to download and extract the datasets are located in the `Model-References/TensorFlow/nlp/bert/data_preprocessing`folder. But the default method in Habana's repository needs to be slighlty modified to run on Voyager.

The yaml file `data_gen.yaml` in our folder can be executed with `kubectl` to download and extract the data. It takes several hous and 100GB+ of data. Follow the followin steps to succesfully run it.

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

An additional file needs to be modified it the Habana's repository. The file is `Model-References/TensorFlow/nlp/bert/data_preprocessing/WikiDownloader.py` and the command `lbzip2` should be `bzip2` (because the package is not installed by default in Ubuntu). **Optional**: The `bzip2` unpacking may take some time (1+ hour) so better to move that line to inside the if statement if you neeed to run it again.


Beofre launching the pod take into account the following. The `yaml` file assumes the repository is located in:
```bash
cd /home/youruser/models/Model-References
```
so modify it to your own folder. Also the `Ceph` should be mounted to your own folder.

Finally, to launch the pod:

```bash
kubectl create -f data_gen.yaml 
```
