# BART model using PyTorch and simpletransformers
This folder contains the scripts and instructions to fine-tune the BART model on Voyager. The model is mantained by Habana and you can find it in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/nlp/BART/simpletransformers) for a deeper explanation of the model.
 
The model was verified on Voyager with SynapseAI version 1.13.



## Download pre-training dataset

The following commands can be used to obtain the pre-trained model.
```bash
mkdir data
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz -P data
tar -xvf data/paws_wiki_labeled_final.tar.gz -C data
mv data/final/* data
rm -r data/final

wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -P data
``` 
This will dowload the datasert to `./data` folder. Remember to use this folder when runinng the model.


**NOTE:**Alternatively, you can copy the dataset from:

```bash
/voyager/ceph/users/javierhn/datasets/BART/data
```

## Training
### Single Card

- The fine-tuning of the BART model can be done using the `bart_1card.yaml` in this folder. The model runs using BF16 mixed precision. Launch the pod with
  ```bash
  kubectl create -f bart_1card.yaml
  ``` 
  whill will execute the following:
  ```bash
  hl-smi;
  export HOME=/scratch/tmp;
  mkdir -p /scratch/tmp/;
  cd /scratch;
  git clone -b 1.13.0 https://github.com/HabanaAI/Model-References;
  export PYTHONPATH=/scratch/Model-References:/scratch/Model-References/PyTorch/nlp/BART/simpletransformers:$PYTHONPATH;
  cd Model-References/PyTorch/nlp/BART/simpletransformers;
  pip install -e .;
  pip install bert_score;
  python3 examples/seq2seq/paraphrasing/train.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --save_best_model --data_dir /dataset --output_dir /output --bf16 autocast
  ``` 
  Remember to change the *dataset* and *output* volumes to your own. The Single card run is too slow and runs out of memory after a few epochs but it can be used to test the model.


### 8 Cards (single node)

Most of the parameters passed to the model are the same as in a single card. But the *yaml* files have been modified to run in multiple HPUs. Here we launch an MPIJob to run the model in one node (with 8 cards). 

- Fine-tuning of BART on 8HPUs, BF16, batch size 32, Lazy mode. The MPIJob can be launched using the `bart_8cards.yaml` located in this folder. You need to modify the `mydir` variable to point to the location of your `yaml` and `setup.sh` files. 
 
  Remember to change the *dataset* and *output* folders to your own in BOTH launcher and worker.

### 16 cards (multi-node)

You can find in `bart_8cards.yaml` and emmaple on how to run in multiple nodes. Note that it only has minor modifications from the `8cards` case.


