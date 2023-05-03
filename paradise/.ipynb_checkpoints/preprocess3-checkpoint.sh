TGT_LANG=$1
AFROMT_PATH=$2

for i in train valid test; do for j in ln ; do python /home/jupyter/afromt/training/scripts/spm_encode.py --model=$AFROMT_PATH/spm_model.model < $i.$j > $i.spm.$j & done; done

fairseq-preprocess \
	--trainpref train.spm --validpref valid.spm --testpref test.spm --destdir data-bin --workers=80 --source-lang ln --only-source --bpe sentencepiece --srcdict $AFROMT_PATH/fairseq.vocab 

