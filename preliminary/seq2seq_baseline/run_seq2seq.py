# !/usr/bin/python3
# -*- coding: UTF-8 -*-
import sys
from seq2seq_baseline import execute

if __name__ == '__main__':
    mode = sys.argv[1]
    print('\n>> Mode : %s\n' % mode)

    if mode == 'train':
        # start training
        execute.train()
    elif mode == 'test':
        # interactive decode
        execute.decode()


