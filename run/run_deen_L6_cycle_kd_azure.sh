echo "======================= GPU & CUDA Version Checks ========================"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V

echo "=================== Python & TensorFlow Version Checks ==================="
python -V
python -c "import torch; print(torch.__version__)" 
echo "PHILLY_GPU_COUNT" $PHILLY_GPU_COUNT
pwd

echo "======================= Philly File System Checks ========================"
echo -n "CURRENT_DIRECTORY "
pwd
echo "PHILLY_HOME" $PHILLY_HOME
ls -alh $PHILLY_HOME
echo "PHILLY_USER_DIRECTORY" $PHILLY_USER_DIRECTORY
ls -alh $PHILLY_USER_DIRECTORY

df -h


echo "ALL ARGS" $@

model=transformer
PROBLEM=IWSLT14_DE_EN
ARCH=transformer_iwslt_de_en_L6
CYCLE_STEP=${3:-12000}
TEMPERATURE=${4:-1.0}
TRADEOFF=${5:-0.5}

SETTING=cycle${CYCLE_STEP}_T${TEMPERATURE}_tradeoff${TRADEOFF}


CODE_PATH=/hdfs/msrmt/fetia/fairseq
REMOTE_DATA_PATH=/hdfs/msrmt/fetia/data/fairseq
REMOTE_MODEL_PATH=/hdfs/msrmt/fetia/ckpts/fairseq

NUMBER_DEVICE=4

nvidia-smi

mkdir -p ${REMOTE_MODEL_PATH}/$model/$PROBLEM/${ARCH}_${SETTING}

CUDA_VISIBLE_DEVICES=0 python ${CODE_PATH}/train.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en \
	--arch $ARCH --max-update 110000 --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler cosine  --cosine-cycle-steps ${CYCLE_STEP} --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.001 --min-lr 1e-09  \
  --weight-decay 0.0 --criterion kd_cross_entropy --label-smoothing 0.1 \
  --kd-trade-off ${TRADEOFF} --start-ensemble-training-cycle 3 --teachers-cnt 3 --kd-temperature ${TEMPERATURE} \
  --max-tokens 4096 --no-progress-bar --save-dir ${REMOTE_MODEL_PATH}/$model/$PROBLEM/${ARCH}_${SETTING} | tee ${CODE_PATH}/log/${model}_${PROBLEM}_${ARCH}_${SETTING}.txt
   

