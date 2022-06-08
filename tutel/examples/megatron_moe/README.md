# Training Bert on Megatron_LM with Tutel:
## Install Tutel
```shell
git clone https://github.com/microsoft/tutel --branch main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py
```

## Get and install Megatron_LM
```shell
cd ./tutel/tutel/examples/megatron_moe
git clone https://github.com/NVIDIA/Megatron-LM --branch main
cd Megatron-LM && git checkout d898a8991d1a08d29074f87819d1bf41517e35f5
# This patch is an example to train Fairseq MoE transformers.
# Note that the current patch only works for `legacy_ddp` backend, and `--checkpoint-activations` must be disabled.
git apply ../megatron_patch.diff
# python3 -m pip install --no-deps --editable .
```

## Prepare the dataset
Download public data, for preprocessing own dataset, please refer [Megatron_LM](https://github.com/NVIDIA/Megatron-LM#data-preprocessing).
```shell
mkdir data && cd data
```
Use HackerNewsDataset:
```shell
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/HackerNewsDataset_text_document.bin
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/HackerNewsDataset_text_document.idx
cd ..
```
Use WikipediaDataset:
```shell
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/WikipediaDataset_text_document.bin
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/WikipediaDataset_text_document.idx
cd ..
```


## Train a Bert model with Tutel MoE (MOE is moe-freq)
```shell
mkdir -p ./checkpoints
CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=../../bert-vocab.txt
DATA_PATH=./data

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters 2000000 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
	   --micro-batch-size 4 \
           --global-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --activations-checkpoint-method uniform"

python pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH

```