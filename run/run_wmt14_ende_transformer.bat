model=transformer
PROBLEM=WMT14_ENDE
SETTING=transformer_vaswani_wmt_en_de_big

REMOTE_DATA_PATH=/hdfs/sdrgvc/fetia/transformer_pytorch_data
REMOTE_MODEL_PATH=/hdfs/sdrgvc/fetia/transformer_pytorch_model
CODE_PATH=/var/storage/shared/sdrgvc/fetia/fairseq

python ../train.py F:\Data_Models\NMT\wmt14_en_de_joined_dict --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09  --update-freq 16 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --no-progress-bar --save-dir F:\Data_Models\NMT\ckpts\WMT14_EN_DE
	

