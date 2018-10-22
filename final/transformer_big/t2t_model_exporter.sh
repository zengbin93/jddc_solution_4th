#! /bin/bash

export PATH=/usr/local/python3.6.5/bin/:$PATH

DATA_DIR=/home/team55/notespace/data/t2t_big/data
TRAIN_DIR=/home/team55/notespace/data/t2t_big/model
TMP_DIR=/home/team55/notespace/data/t2t_big/tmp

# mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

USR_DIR=.
PROBLEM=jddc_big
MODEL=transformer
HPARAMS=transformer_big

t2t-exporter \
  --t2t_usr_dir=$USR_DIR \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR