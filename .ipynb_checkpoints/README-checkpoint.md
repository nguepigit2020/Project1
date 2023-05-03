# Pretraining Strategies using Monolingual and Parallel Data for Low-Resource Machine Translation

Code for the AMMI Final project
## Clone the repo

```bash
git clone https://github.com/nguepigit2020/Project1.git
```
## Data

Please use `gdown` to download the data
```bash
cd Project1
gdown --id 1otanXj-hIVPLG8xFBjPPsF4yLE3wquvM
tar -xf data.tar.xz
```
## Installation
Run the following commands
```bash
cd training
# install apex for fp16 for faster training
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
# install fairseq
pip install -e .
cd ../
```
## Preprocessing
### For parallel data
First be sure to install [sentencepiece](https://github.com/google/sentencepiece) 
```bash
TGT_LANG='ln'#whatever target language you choose
cd Project1 # the data folder
cd en-$TGT_LANG
bash ../preprocess.sh $TGT_LANG ../paradise/data-bin/translation/en-$TGT_LANG #Do this for all parallel data
cd ../../
```
### For monolingual data
```bash
SRC_LANG='en'#whatever target language you choose
cd Project1 # the data folder
cd monolingual_folder #The folder where you monolingual is store
bash ../../preprocess3.sh $SRC_LANG ../paradise/data-bin/denoising/monolingual_folder
cd ../../
```
## Pretraining
```bash
cd paradise
bash pretraining.sh
```
## Fine tuning
```bash
bash train_paradise.sh Project1/en-$TGT_LANG/data-bin/translation/ Project/en-$TGT_LANG/model_output $TGT_LANG
```
## Evaluation
Once training is over, you can generate from the model as follows:
```bash
fairseq-generate Project1/en-$TGT_LANG/data-bin/translation/ --gen-subset test --path Project1/en-$TGT_LANG/model_outputs/checkpoints/checkpoint_best.pt  --beam 5 --batch-size 300 --remove-bpe sentencepiece --truncate-source --task translation_from_xbart -s en -t $TGT_LANG --lenpen $LENPEN > generation.log 
```
and generate BLEU scores as follows:
```bash
bash eval_bleu_chrf.sh generation.log
```
https://drive.google.com/file/d/1d4P1RXin322Sr4W2rYeGsBXUW5UfWCZJ
## Fine turning directly from pretraining model

### Model usage
Download the model as follows:
```bash
gdown --id 1d4P1RXin322Sr4W2rYeGsBXUW5UfWCZJ
# and you will get checkpoint_best.pt which is the model file
```

### Training
```bash
bash train_paradise.sh Project1/en-$TGT_LANG/data-bin/translation/ Project/en-$TGT_LANG/model_output $TGT_LANG
```
and you should see your model training!


## The script code will be written clearly very soon.
