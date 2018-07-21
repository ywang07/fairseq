model=transformer
PROBLEM=IWSLT14_DEEN
ARCH=transformer_iwslt_de_en_L4
SETTING=vanilla

mkdir -p $model/$PROBLEM/${ARCH}_${SETTING}

CUDA_VISIBLE_DEVICES=0 python train.py /home/fetia/transformer_pytorch/Transformer-PyTorch-master/data-bin/iwslt14.tokenized.de-en \
	--clip-norm 0.0 --max-tokens 4000 --share-decoder-input-output-embed \
	--arch $ARCH --save-dir $model/$PROBLEM/${ARCH}_${SETTING} \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.001 --warmup-init-lr 1e-07 \
	--warmup-updates 4000 --max-update 200000
