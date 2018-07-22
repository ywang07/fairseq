python train.py F:\Data_Models\NMT\iwslt14.tokenized.de-en --clip-norm 0.0 --min-lr 1e-09  --max-tokens 4096 --share-decoder-input-output-embed --arch transformer_iwslt_de_en_L6 --save-dir F:\Data_Models\NMT\ckpts\IWSLT14_DE_EN_L6 --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 --lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --lr 0.001 --warmup-init-lr 1e-07 --warmup-updates 4000 --max-update 160000  
rem >./log/IWSLT14_DEEN_dim256.txt 2>&1
rem >./log/IWSLT14_DEEN_dim256.txt
