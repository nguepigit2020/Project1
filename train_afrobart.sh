# langs="en,xh,af,zu,sw,bem,run,ln"
# DATA=$1 # path to data-bin-afromt
# OUTDIR=$2 # path to model logging/checkpointing dir
# TGT_LANG=$3
langs="en,ln,zu,af,sw,"
DATA=/home/jupyter/afromt/en-zu/data-bin-afromt
OUTDIR=/home/jupyter/afromt/paradise/model_output_which_en_zu
TGT_LANG='zu'

mkdir -p $OUTDIR/log
# PRETRAIN=afrobart.pt
PRETRAIN=/home/jupyter/afromt/paradise/pretrainining_output_new/checkpoint_best.pt
# PRETRAIN=afromt/mbart.cc25.v2/model.pt # change if need be


fairseq-train $DATA --fp16 \
--arch mbart_base --layernorm-embedding \
--task translation_from_pretrained_bart \
--source-lang en --target-lang $TGT_LANG \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-05 --stop-min-lr -1 --warmup-updates 5000 --total-num-update 50000  \
--eval-bleu --eval-bleu-remove-bpe \
--dropout 0.3 --weight-decay 0.01 --attention-dropout 0.1 \
--max-tokens 4096 --update-freq 2 --max-source-positions 1024 --max-target-positions 1024 \
--validate-interval=10 --save-interval=10 --save-interval-updates 1000 --keep-interval-updates 2 --no-epoch-checkpoints --validate-after-updates 2500 \
--seed 666 --log-format simple --log-interval 20 \
--save-dir $OUTDIR/checkpoints \
--skip-invalid-size-inputs-valid-test --tensorboard-logdir $OUTDIR/tensorboard \
--decoder-embed-dim=768 \
--decoder-ffn-embed-dim=3072 \
--restore-file $PRETRAIN --save-dir $OUTDIR/checkpoints \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
--encoder-embed-dim=768 \
--encoder-ffn-embed-dim=3072 \
--encoder-layers=6 \
--langs $langs \
--encoder-attention-heads=12 \
--decoder-layers=6 --decoder-attention-heads=12 \
--ddp-backend=no_c10d | tee -a $OUTDIR/train.log
