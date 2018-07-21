#NAME=$9
model=transformer
PROBLEM=WMT14_ENDE
SETTING=transformer_vaswani_wmt_en_de_big

REMOTE_DATA_PATH=/hdfs/sdrgvc/fetia/transformer_pytorch_data
CODE_PATH=/var/storage/shared/sdrgvc/fetia/fairseq
REMOTE_MODEL_PATH=/hdfs/sdrgvc/fetia/transformer_pytorch_model

nvidia-smi

python -c "import torch; print(torch.__version__)"



python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/wmt14_en_de_joined_dict --path ${REMOTE_MODEL_PATH}/${model}/${PROBLEM}/${SETTING}/checkpoint_best.pt --batch-size 128 --beam 4 --lenpen 0.6 --quiet --remove-bpe --no-progress-bar