#! /bin/bash

export PATH=/usr/local/python3.6.5/bin/:$PATH

BASE_DIR=/home/team55/notespace/data/t2t_big_rq
# BASE_DIR=/submitwork/t2t
DATA_DIR=$BASE_DIR"/data"
TRAIN_DIR=$BASE_DIR"/model"
TMP_DIR=$BASE_DIR"/tmp"
# CHECKPOINT_PATH=$TRAIN_DIR"/model.ckpt-48000"
CHECKPOINT_PATH=$BASE_DIR"/avg_5_39000/model.ckpt-45000"

USR_DIR=/home/team55/notespace/zengbin/transformer_big_rq/
# USR_DIR=/submitwork/transformer_big/
PROBLEM=jddc_big
MODEL=transformer
HPARAMS=transformer_big

# DECODE_FILE=$USR_DIR"test_q.txt"
# OUTPUT_FILE=$USR_DIR"answers.txt"

/usr/local/python3.6.5/bin//t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --decode_interactive=True \
  --hparams_set=$HPARAMS \
  --checkpoint_path=$CHECKPOINT_PATH \
  --decode_hparams="beam_size=4,alpha=0.6,batch_size=32,guess_and_check_top_k=5,write_beam_scores=True,return_beams=True" \



#   --decode_hparams="beam_size=4,alpha=0.6,batch_size=32" \
#   --decode_from_file=$DECODE_FILE \
#   --decode_to_file=$OUTPUT_FILE