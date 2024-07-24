# BERT model using PyTorch
Here we provide the scripts and files to download the dataset and train a BERT model on Voyager using Pytorch. The model is mantained by Habana and you can find it in their [repository](https://github.com/HabanaAI/Model-References/tree/1.13.0/PyTorch/nlp/bert). Please check their repository for a deeper explanation of the BERT model.

The model was verified on Voyager with SynapseAI version 1.15.1.

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google. We are showing here the the BERT-LARGE model with  24-layers, 1024-hidden, 16-heads and 340M parameters which is trained onversion of the English Wikipedia with 2,500M words. The modes was ported by [Habana](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/nlp/bert) but the pretrained was based on [https://github.com/NVIDIA/DeepLearningExamples.git](https://github.com/NVIDIA/DeepLearningExamples.git) and fine-tuning is based on [https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git).

In this folder you will find:
- The scripts and commands you need to get the dataset:
  - The vocab file.
  - Wiki dataset (BookCorpus optional).
  - SQUAD dataset.
  - Pretrained model.
- The yaml files to run:
  - Pretraining of phase1 and phase 2 in a single or multiple HPU cards.
  - Fine tuning (using SQUAD) in one or multiple HPUs.
  - Inference in a single HPU.

## Dataset preparation
Ideally the dataset consists on the Wikiepdia and BookCorpus datasets but by default, following Habana's suggestions, we skip the BookCorpus as it gets many errors and broken links (although this may affect final accuracy). It takes 20h+ and 200Gb+ to download and process the dataset. You can eithier use the dataset which is already in Voyager at:
```bash
/voyager/ceph/users/javierhn/datasets/bert/pytorch
```
or follow the next steps to generate your own dataset.

### Vocab File
You can download the Vocab file using:
```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
```
then you can `unzip` it to extract the files.

### Download and process wikicorpus
You need to clone Habana's repository to a folder in Voyager (for example in `/ceph/temp/`). The following steps will then genereate the dataset in `/ceph/temp/Model-References/PyTorch/nlp/bert/data`. You may then later, at the end of the process,  move the dataset to other folder more convenient for you.

To clone the repository:
```bash
git clone -b 1.11.0 https://github.com/HabanaAI/Model-References
```

You can then use the `data_gen.yaml`, located in this folder, to launch a pod as
```bash
kubectl create -f data_gen.yaml
```
which will execute the following:
```bash
export HOME=/scratch/tmp;
mkdir -p /scratch/tmp/;
cd /ceph/temp;
export PYTHON=python3;
export PYTHONPATH=/ceph/temp/Model-References:/ceph/temp/Model-References/PyTorch/nlp/bert/:$PYTHONPATH;
cd Model-References/PyTorch/nlp/bert;
$PYTHON -m pip install -r requirements.txt;
pip install ipdb nltk progressbar html2text;
cd data;
pip install h5py boto3==1.26.75;
bash create_datasets_from_start.sh;
```
Note that it will only download the Wikipedia dataset and **not** the BookCorpus.

### Finetuning
To extract and pre-process the Squad Dataset(V1.1). Go to
```bash
cd Model-References/PyTorch/nlp/bert/data/squad
```
and execute the following
```bash
bash squad_download.sh
```

### Pretrained model
We are using the checkpoint from a pretrained model for the fine-tuning and inference examples. This checkpoint has been trained by Intel Habana and can be downloaded by executing the following:
```bash
wget https://vault.habana.ai/artifactory/pretrained-models/checkpoints/1.13.0/PT/BertL-PT/BertL-PT-PyTorch-2.1.0-1.13.0-463-32n-checkpoint.tar.gz
``` 


## Training

Before training, you need to create a log directory to store `dlloger.json`. Then, modify the variable `dllog` in these `yaml` files which will point to that log directory. Same for the `output` folder.

All the pre-training examples shown here run a few timesteps to test the model. So the accuracy of the results may be very small.

### Pre-training
**Single Card**

- The pretraining of **Phase 1** of the BERT Large can be done using the `bert_phase1_1card.yaml` in the *1card* folder. Launch the pod with
  ```bash
  kubectl create -f bert_phase1_1card.yaml
  ```   
  Remeber to change the *dataset, dllog* and *output* folders to your own folder. The Single card run is too slow but can be used to test the model.

  It runs in lazy mode 1 HPU, unpacked data, BF16 mixed precision, batch size 64 for Phase1 and a few timesteps only:
  ```bash
  export HOME=/scratch/tmp;
  export PATH=/scratch/tmp/.local/bin:$PATH;
  mkdir -p /scratch/tmp/;
  cd /scratch;
  git clone -b 1.15.1 https://github.com/HabanaAI/Model-References;
  export PYTHONPATH=/scratch/Model-References:$PYTHONPATH;
  cd Model-References/PyTorch/nlp/bert;
  pip install -r requirements.txt;
  python3 run_pretraining.py --do_train --bert_model=bert-large-uncased \
         --autocast --config_file=./bert_config.json \
         --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
         --json-summary=/dllog/dllogger.json --output_dir=/output --use_fused_lamb \
         --input_dir=/dataset/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en \
         --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=10 \
         --warmup_proportion=0.2843 --num_steps_per_checkpoint=5 --learning_rate=0.006 --gradient_accumulation_steps=128 \
         --enable_packed_data_mode False ;

  ```

- **Phase 2** has not been checked in a single card because Phase 1 is too slow.

**8 Cards**

Many parameters passed to the model are the same as in a single card. But the *yaml* files have been modified to run in multiple HPUs. Here we launch an MPIJob to run the model in one node (with 8 cards).

- Pretraining **Phase 1** of BERT Large (Lazy mode, 8 HPUs, unpacked data, FP32, per chip batch size of 64). The MPIJob can be launched using the `bert_phase1_8cards.yaml` located in the `8 cards` folder. You need to modify the `mydir` volumes to point to the location of your `yaml` and `setup.sh` files.

  Remeber to change the *dataset, dllog* and *output* folders to your own. In the launcher AND worker.

  ```bash
  python3 $MODEL_PATH/run_pretraining.py \
              --do_train \
              --bert_model=bert-large-uncased \
              --config_file=$MODEL_PATH/bert_config.json \
              --use_habana \
              --allreduce_post_accumulation \
              --allreduce_post_accumulation_fp16 \
              --json-summary=/dllog/dllogger.json \
              --output_dir=/output \
              --use_fused_lamb \
              --input_dir=/dataset//hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en \
              --train_batch_size=8192 \
              --max_seq_length=128 \
              --max_predictions_per_seq=20 \
              --warmup_proportion=0.2843 \
              --max_steps=30 \
              --num_steps_per_checkpoint=15 \
              --learning_rate=0.006 \
              --gradient_accumulation_steps=128 \
              --enable_packed_data_mode False
  ```

- Pretraining of **Phase 2**. You can use `bert_phase2_8cards.yaml` to launch the MPIJob to train the model. It takes as input the checkpoint from phase 1 (you need the pass the last timestep form Phase 1 in `phase1_end_step`).

  ```bash
  python3 $MODEL_PATH/run_pretraining.py \
            --do_train \
            --bert_model=bert-large-uncased \
            --config_file=$MODEL_PATH/bert_config.json \
            --use_habana \
            --allreduce_post_accumulation \
            --allreduce_post_accumulation_fp16 \
            --json-summary=/dllog/dllogger.json \
            --output_dir=/output \
            --use_fused_lamb \
            --input_dir=/dataset/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en \
            --train_batch_size=4096 \
            --max_seq_length=512 \
            --max_predictions_per_seq=80 \
            --warmup_proportion=0.128 \
            --max_steps=10 \
            --num_steps_per_checkpoint=5 \
            --learning_rate=0.004 \
            --gradient_accumulation_steps=512 \
            --resume_from_checkpoint \
            --phase1_end_step=15 \
            --phase2 \
            --enable_packed_data_mode False
  ```

**16 Cards (multi-node)**

The model can run in multiple nodes. Here we provide a `yaml` file example that runs in 2 nodes (16 HPUs). The model parameters are exactly the same as in the `8cards` case. You can find, in the `16cards` folder, the `bert_phase1_16cards.yaml` and `bert_phase2_16cards.yaml` to run Phase1 and Phase2 respectively.


## Fine-tuning
**8 cards**

You can use the `bert_ftuning_8cards.yaml` to perform the fine-tuning on the pretrained data. We have used Habana's pretrained model as our initial checkpoint. Please make sure that `--init_checkpoint` points to the checkpoint you want. The fine-tuning just requires a few minutes.

```bash
python3 $MODEL_PATH/run_squad.py \
           --do_train \
           --bert_model=bert-large-uncased \
           --config_file=$MODEL_PATH/bert_config.json \
           --use_habana \
           --autocast \
           --use_fused_adam \
           --do_lower_case \
           --output_dir=/output \
           --json-summary=/dllog/dllogger.json \
           --train_batch_size=24 \
           --predict_batch_size=8 \
           --seed=1 \
           --max_seq_length=384 \
           --doc_stride=128 \
           --max_steps=-1  \
           --learning_rate=3e-5 \
           --num_train_epochs=2 \
           --init_checkpoint=/dataset/pretrained-habana/1.13/ckpt_7038.pt \
           --vocab_file=/dataset/uncased_L-24_H-1024_A-16/vocab.txt \
           --train_file=/dataset/squad/v1.1/train-v1.1.json \
           --skip_cache \
           --do_predict  \
           --predict_file=/dataset/squad/v1.1/dev-v1.1.json \
           --do_eval \
           --eval_script=/dataset/squad/v1.1/evaluate-v1.1.py \
           --log_freq 20 

```

## Inference
Inference can be run in a single HPU. To run it, one can use the `bert_inference_1card.yaml` file. You need to modify the environment variable `vocab` to the path you store it. Similarly, the `init_ckpt` should point to the folder with your last checkpoint (we have used Habana's pretrained model).

It will execute: 
```bash
python3 run_squad.py --bert_model=bert-large-uncased --autocast \
           --config_file=./bert_config.json \
           --use_habana --do_lower_case --output_dir=/output \
           --json-summary=/dllog/dllogger.json \
           --predict_batch_size=24 \
           --init_checkpoint=/dataset/pretrained-habana/1.13/ckpt_7038.pt \
           --vocab_file=/dataset/uncased_L-24_H-1024_A-16/vocab.txt \
           --do_predict  \
           --predict_file=/dataset/squad/v1.1/dev-v1.1.json \
           --do_eval --eval_script=/dataset/squad/v1.1/evaluate-v1.1.py

```
 
