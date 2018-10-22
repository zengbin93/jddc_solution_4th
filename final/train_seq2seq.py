#! /usr/local/python3.6.5/bin//python3.6
# -*- coding: utf-8 -*-

import os
import torch

from jddc.seq2seq.fields import *
from jddc.seq2seq.optim import Optimizer
from jddc.seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from jddc.seq2seq.loss import NLLLoss
from jddc.seq2seq.supervised_trainer import SupervisedTrainer
import jddc.utils as u


# solution for  _csv.Error: field larger than field limit (131072)
import csv
csv.field_size_limit(500 * 1024 * 1024)


class Seq2SeqConfig(object):
    """Seq2Seq模型参数配置"""
    use_cuda = True
    teacher_forcing_ratio = 0.5

    # encoder & decoder
    hidden_size = 256
    n_layers = 5
    bidirectional = True
    max_len = 200
    rnn_cell = 'lstm'

    encoder_params = u.AttrDict()
    encoder_params['hidden_size'] = hidden_size
    encoder_params['n_layers'] = n_layers
    encoder_params['bidirectional'] = bidirectional
    encoder_params['max_len'] = max_len
    encoder_params['rnn_cell'] = rnn_cell
    encoder_params['variable_lengths'] = True
    encoder_params['input_dropout_p'] = 0
    encoder_params['dropout_p'] = 0.3

    decoder_params = u.AttrDict()
    decoder_params['hidden_size'] = hidden_size*2 if bidirectional else hidden_size
    decoder_params['n_layers'] = n_layers
    decoder_params['bidirectional'] = bidirectional
    decoder_params['max_len'] = max_len
    decoder_params['rnn_cell'] = rnn_cell
    decoder_params['use_attention'] = True
    decoder_params['use_cuda'] = use_cuda
    decoder_params['input_dropout_p'] = 0
    decoder_params['dropout_p'] = 0.3

    def __init__(self):
        # 模型存储目录
        self.s2s_path = os.path.join("/home/team55/notespace/data", "seq2seq")
        u.insure_folder_exists(self.s2s_path)
        self.file_train = os.path.join(self.s2s_path, "train.tsv")
        # 翻转QQ分词结果
        self.file_train_rq = os.path.join(self.s2s_path, "train_reverse_q.tsv")
        self.log_file = os.path.join(self.s2s_path, "seq2seq_02.log")


conf = Seq2SeqConfig()

logger = u.create_logger(name='seq2seq', log_file=conf.log_file, cmd=True)


# 加载数据集
# --------------------------------------------------------------------
print("loading dataset ...")
src = SourceField(batch_first=True)
tgt = TargetField(batch_first=True)
max_len = conf.max_len


def len_filter(example):
    return len(example.src) <= 100 and len(example.tgt) <= 100


train = torchtext.data.TabularDataset(
    path=conf.file_train_rq, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

src.build_vocab(train, max_size=200000)
tgt.build_vocab(train, max_size=100000)
input_vocab = src.vocab
output_vocab = tgt.vocab

# define model
loss = NLLLoss()
logger.info(str(conf.encoder_params))
logger.info(str(conf.decoder_params))
encoder = EncoderRNN(vocab_size=len(src.vocab), **conf.encoder_params)
decoder = DecoderRNN(vocab_size=len(tgt.vocab), eos_id=tgt.eos_id,
                     sos_id=tgt.sos_id, **conf.decoder_params)
seq2seq = Seq2seq(encoder, decoder)

optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
# scheduler = StepLR(optimizer.optimizer, 1)
# optimizer.set_scheduler(scheduler)

for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)

if conf.use_cuda:
    seq2seq.cuda()
    loss.cuda()

# train
trainer = SupervisedTrainer(loss=loss, batch_size=50,
                            checkpoint_every=500, print_every=10,
                            expt_dir=conf.s2s_path,
                            random_seed="1234", use_cuda=conf.use_cuda)

trainer.logger = logger

seq2seq = trainer.train(seq2seq, train, num_epochs=5, optimizer=optimizer,
                        teacher_forcing_ratio=0.5, resume=True)
