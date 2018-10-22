# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import codecs
import logging
import traceback
from zb.tools.logger import create_logger

from seq2seq import data_utils
from seq2seq.seq2seq_model import Seq2SeqModel
from seq2seq.config import BaseConf, TestConf, TrainConf

base_conf = BaseConf()

log_file = os.path.join(base_conf.log_path, 'execute.log')
logger = create_logger(log_file, name='exec', cmd=True)

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# (source_size, target_size)
_buckets = [(50, 50), (80, 60), (150, 70), (500, 90)]


def empty_file(file):
    with codecs.open(file, mode='w', encoding='utf-8') as f:
        f.truncate()

def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.

  Notes:
      读取文件是source和target文件一起读的，每一次读操作都是读一个sentence pair（一句来自
      source，一句来自target），读取之后根据长度将该pair装入到相应的桶里。
  """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 10000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only, model_path=None):
    """Create model and initialize or load parameters"""
    model = Seq2SeqModel(base_conf.enc_vocab_size,
                        base_conf.dec_vocab_size,
                        _buckets,
                        base_conf.layer_size,
                        base_conf.num_layers,
                        base_conf.max_gradient_norm,
                        base_conf.batch_size,
                        base_conf.learning_rate,
                        base_conf.learning_rate_decay_factor,
                        forward_only=forward_only,
                        use_lstm=base_conf.use_lstm)
    if model_path:
        logger.info("Reading model parameters from %s" % model_path)
        model.saver.restore(session, model_path)
        return model
    else:
        # load pre_model or not
        ckpt = tf.train.get_checkpoint_state(base_conf.work_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model


def train():
    train_conf = TrainConf()
    log_file = os.path.join(train_conf.log_path, 'train.log')
    logger = create_logger(log_file, name='Train', cmd=True)
    logger.info("start training")

    # prepare dataset
    logger.info("prepare dataset ...")
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(
        train_conf.work_path, train_conf.train_enc, train_conf.train_dec,
        train_conf.dev_enc,
        train_conf.dev_dec, train_conf.enc_vocab_size, train_conf.dec_vocab_size)
    logger.info("dataset prepared!")

    logger.info("enc_train: %s; dec_train: %s;" % (enc_train, dec_train))
    logger.info("enc_dev: %s; dec_dev: %s;" % (enc_dev, dec_dev))

    # setup config to use BFC allocator
    config = tf.ConfigProto(
        device_count={"CPU": 8},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )
    config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=config) as sess:
        # Create model.
        logger.info("Creating %d layers of %d units." % (train_conf.num_layers, train_conf.layer_size))
        model = create_model(sess, False)

        logger.info("Read data into buckets and compute their sizes.")
        dev_set = read_data(enc_dev, dec_dev)
        train_set = read_data(enc_train, dec_train, train_conf.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / train_conf.steps_per_checkpoint
            loss += step_loss / train_conf.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % train_conf.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                logger.info("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss. save the model!!
                checkpoint_path = os.path.join(train_conf.work_path, "seq2seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        logger.info("  eval: empty bucket %d" % bucket_id)
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    logger.info("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


def decode():
    test_conf = TestConf()
    with tf.Session() as sess:
        # Create model structure and load parameters
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        enc_vocab_path = os.path.join(test_conf.work_path, "vocab%d.enc" % test_conf.enc_vocab_size)
        dec_vocab_path = os.path.join(test_conf.work_path, "vocab%d.dec" % test_conf.dec_vocab_size)

        enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

        # Decode from standard input.
        test_path = test_conf.test
        result_path = test_conf.result
        empty_file(result_path)
        with codecs.open(test_path, mode='r', encoding='utf-8') as rf:
            with codecs.open(result_path, mode='a', encoding='utf-8') as wf:
                try:
                    sentence = rf.readline()
                    while sentence:
                        sentence = sentence.rstrip('<s>')
                        # Get token-ids for the input sentence.

                        token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
                        # Which bucket does it belong to?
                        bucket_id = min([b for b in range(len(_buckets))
                                         if _buckets[b][0] > len(token_ids)])
                        # Get a 1-element batch to feed the sentence to the model.
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            {bucket_id: [(token_ids, [])]}, bucket_id)

                        """logits可以理解成未进入softmax的概率，一般是输出层的输出，softmax的输入"""
                        # Get output logits for the sentence.
                        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                         target_weights, bucket_id, True)
                        # This is a greedy decoder - outputs are just argmaxes of output_logits.
                        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                        # If there is an EOS symbol in outputs, cut them at that point.
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        # Print out French sentence corresponding to outputs.（corresponding to:与...一致...）
                        result = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs if
                                          tf.compat.as_str(rev_dec_vocab[output]) not in [",", "_UNK"]])
                        wf.write(result + '\n')
                        sentence = rf.readline()
                except Exception as e:
                    traceback.print_exc()
                    logging.error("test failure", e)


def run_prediction(input_file_path, output_file_path):
    log_file = os.path.join(base_conf.log_path, 'prediction.log')
    logger = create_logger(log_file, name='predictor', cmd=True)
    logger.info('run prediction ...')
    test_conf = TestConf()
    with tf.Session() as sess:
        # Create model structure and load parameters
        model = create_model(sess, True, model_path=test_conf.model)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        enc_vocab_path = os.path.join(test_conf.work_path, "vocab%d.enc" % test_conf.enc_vocab_size)
        dec_vocab_path = os.path.join(test_conf.work_path, "vocab%d.dec" % test_conf.dec_vocab_size)

        enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

        # Decode from standard input.
        test_path = input_file_path
        result_path = output_file_path
        empty_file(result_path)
        with codecs.open(test_path, mode='r', encoding='utf-8') as rf:
            with codecs.open(result_path, mode='a', encoding='utf-8') as wf:
                try:
                    sentence = rf.readline()
                    while sentence:
                        sentence = sentence.rstrip('<s>')
                        # Get token-ids for the input sentence.
                        logger.info("current sentence: " + sentence)
                        token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
                        logger.info("token_ids: " + ' '.join([str(i) for i in token_ids]))
                        # Which bucket does it belong to?
                        bucket_id = min([b for b in range(len(_buckets))
                                         if _buckets[b][0] > len(token_ids)])
                        logger.info('bucket_id: ' + str(bucket_id))
                        # Get a 1-element batch to feed the sentence to the model.
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            {bucket_id: [(token_ids, [])]}, bucket_id)

                        """logits可以理解成未进入softmax的概率，一般是输出层的输出，softmax的输入"""
                        # Get output logits for the sentence.
                        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                         target_weights, bucket_id, True)
                        # This is a greedy decoder - outputs are just argmaxes of output_logits.
                        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                        # If there is an EOS symbol in outputs, cut them at that point.
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        # Print out French sentence corresponding to outputs.（corresponding to:与...一致...）
                        result = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs if
                                          tf.compat.as_str(rev_dec_vocab[output]) not in [",", "_UNK"]])
                        wf.write(result + '\n')
                        logger.info("result: " + result)
                        sentence = rf.readline()
                except Exception as e:
                    traceback.print_exc()
                    logging.error("run prediction fail:", e)
    logger.info('run prediction end!')


if __name__ == '__main__':
    mode = sys.argv[1]
    print('\n>> Mode : %s\n' % mode)

    if mode == 'train':
        # start training
        train()
    elif mode == 'test':
        # interactive decode
        decode()

