# !/usr/bin/python3
# -*- coding: UTF-8 -*-
import os


class BaseConf(object):
    # Mode : Train, Test
    mode = None
    base_path = "/home/team55/notespace/zengbin/model1_seq2seq"
    # base_path = "/submitwork"

    # vocabulary size
    enc_vocab_size = 6000
    dec_vocab_size = 6000

    # number of LSTM layers : 1/2/3
    num_layers = 3
    use_lstm = True

    # typical options : 128, 256, 512, 1024
    layer_size = 128

    # dataset size limit; typically none : no limit
    max_train_data_size = 0
    batch_size = 64

    # steps per checkpoint
    steps_per_checkpoint = 10

    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0

    # files
    files = {
        "chat": '/mnt/dataset/preliminaryData/chat.txt',
        "ware": '/mnt/dataset/preliminaryData/ware.txt',
        "order": '/mnt/dataset/preliminaryData/order.txt',
        "user": '/mnt/dataset/preliminaryData/user.txt',
        # "questions": '/home/team55/notespace/zengbin/data/questions.txt',
        # "answers": '/home/team55/notespace/zengbin/data/answers.txt',
    }

    def __init__(self):
        self.data_path = os.path.join(self.base_path, 'data')
        self.log_path = os.path.join(self.base_path, 'log')
        # self.work_path = os.path.join(self.base_path, 'work')
        self.work_path = os.path.join(self.base_path, 'work_lstm')


class TrainConf(BaseConf):
    mode = 'Train'

    def __init__(self):
        super().__init__()
        self.train_enc = os.path.join(self.data_path, 'train_questions.txt')
        self.train_dec = os.path.join(self.data_path, 'train_answers.txt')
        self.dev_enc = os.path.join(self.data_path, 'dev_questions.txt')
        self.dev_dec = os.path.join(self.data_path, 'dev_answers.txt')


class TestConf(BaseConf):
    mode = 'Test'
    test = "/mnt/dataset/TestData/questions50.txt"
    model = "/submitwork/work_lstm/seq2seq.ckpt-720"

    def __init__(self):
        super().__init__()
        self.result = os.path.join(self.work_path, 'test_results.txt')


def load_conf(mode='Train'):
    if mode == 'Train':
        conf = TrainConf()
    elif mode == 'Test':
        conf = TestConf()
    else:
        raise ValueError("mode error, must be 'Train' or 'Test'!!!")
    return conf
