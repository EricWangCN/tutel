# Training a GPT model on Megatron_LM with Tutel:
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
For preprocessing own dataset, please refer [Megatron_LM](https://github.com/NVIDIA/Megatron-LM#data-preprocessing).
```shell
mkdir data && cd data
```

Use public Wikipedia Processed Dataset:
```shell
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/WikipediaDataset_text_document.bin
curl -LO https://mystic.the-eye.eu/public/AI/pile_neox/data/WikipediaDataset_text_document.idx
cd ..
```

## Train a GPT model with Naive Megatron MoE
```shell
../run_megatron_gpt_moe_naive.sh
```

## Train a GPT model with Tutel MoE (MOE is moe-freq)
```shell
../run_megatron_gpt.sh
```