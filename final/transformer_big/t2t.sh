#! /bin/bash

export PATH=/usr/local/python3.6.5/bin/:$PATH

DATA_DIR=/home/team55/notespace/data/t2t_big/data
TRAIN_DIR=/home/team55/notespace/data/t2t_big/model
TMP_DIR=/home/team55/notespace/data/t2t_big/tmp

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

USR_DIR=.
PROBLEM=jddc_big
MODEL=transformer
HPARAMS=transformer_big

# generate data
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM


# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=4 \
  --hparams='batch_size=1024' \
  --output_dir=$TRAIN_DIR


