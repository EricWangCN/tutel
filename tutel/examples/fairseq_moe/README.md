# Training WikiText-103 on fairseq with Tutel:
## Install Tutel
```shell
git clone https://github.com/microsoft/tutel --branch main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py
```

## Install fairseq
```shell
git clone https://github.com/facebookresearch/fairseq --branch main
cd fairseq/ && git checkout b5e7b250913120409b872a940fbafec4d43c7b13
# This patch is an example to make Fairseq Legacy DDP use MoE transformer simply.
# The patch replaces all transformer FFN layers into Tutel MoE layers (but load-balance loss is not contributed to model loss).
git apply ../fairseq_full_patch.diff
python3 -m pip install --editable .
```

## Prepare the dataset
Download [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):
```shell
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```
Preprocess the data:
```shell
fairseq-preprocess \
    --only-source \
    --trainpref wikitext-103/wiki.train.tokens \
    --validpref wikitext-103/wiki.valid.tokens \
    --testpref wikitext-103/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

## Train a Model with Tutel moe (moe-freq=N)
```shell
# Need to be modified
$DATA_PATH=../wikitext-103
$MOE_FREQ=1
$SAVE_DIR=../fairseq_checkpoints

# Train on 8GPUs with FP16
MOE=$MOE_FREQ python3 -m torch.distributed.launch --nproc_per_node=8 train.py $DATA_PATH \
    --ddp-backend ${DDP:-legacy_ddp} \
    --checkpoint-activations \
    --fp16 --fp16-init-scale 4 --fp16-no-flatten-grads \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 500000 --log-format json --log-interval 100 \
    --save-dir $SAVE_DIR

# Train on 8GPUs with FP32
MOE=$MOE_FREQ python3 -m torch.distributed.launch --nproc_per_node=8 train.py $DATA_PATH \
    --ddp-backend ${DDP:-legacy_ddp} \
    --checkpoint-activations \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 500000 --log-format json --log-interval 100 \
    --save-dir $SAVE_DIR
```
