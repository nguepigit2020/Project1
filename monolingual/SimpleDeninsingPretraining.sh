################################
DATA=/home/jupyter/afromt/paradise/data-bin/denoising/ln
OUTDIR=/home/jupyter/afromt/monolingual/output_10.04.2023_Second
# PRETRAIN=/home/jupyter/afromt/paradise/model_output_from_afromt/checkpoints/afrobart.pt
mkdir -p $OUTDIR/log

fairseq-train \
$DATA \
--task=denoising \
--tensorboard-logdir=$OUTDIR/log \
--arch=mbart_base \
--attention-dropout=0.1 \
--no-progress-bar \
--criterion=cross_entropy \
--lr-scheduler=polynomial_decay \
--skip-invalid-size-inputs-valid-test \
--update-freq=4 \
--optimizer=adam \
--adam-betas="(0.9, 0.98)" \
--lr=0.0007 \
--warmup-updates=10000 \
--dropout=0.1 \
--weight-decay=0.01 \
--train-subset=train \
--valid-subset=valid \
--max-update=100000 \
--save-dir=$OUTDIR/ \
--mask=0.3 \
--mask-random=0.1 \
--poisson-lambda=3.5 \
--permute-sentences=1 \
--mask-length=span-poisson \
--replace-length=1 \
--max-source-positions=1024 \
--max-target-positions=1024  \
--share-all-embeddings \
--layernorm-embedding \
--log-interval=10 \
--log-format=json \
--seed=1 \
--min-loss-scale=0.0001 \
--clip-norm=0.1 \
--optimizer-overrides={} \
--save-interval-updates=2500 \
--keep-interval-updates=10 \
--validate-interval=25 \
--keep-last-epochs=-1 \
--keep-best-checkpoints=-1 \
--no-epoch-checkpoints \
--best-checkpoint-metric=loss \
--patience=-1 \
--adam-eps=1e-06 \
--power=1 \
--total-num-update=100000 \
--num-workers=30 \
--no-progress-bar \
--sample-break-mode=complete_doc \
--fp16 \
--max-tokens=2500 \
--max-tokens-valid=2500 \
--encoder-embed-dim=768 \
--encoder-ffn-embed-dim=3072 \
--encoder-layers=6 \
--encoder-attention-heads=12 \
--decoder-layers=6 \
--decoder-attention-heads=12 \
--encoder-learned-pos \
--decoder-embed-dim=768 \
--decoder-ffn-embed-dim=3072 \
--decoder-learned-pos \
--decoder-output-dim=768 \
--activation-fn=gelu \
--pooler-activation-fn=tanh \
--tokens-per-sample 512 \