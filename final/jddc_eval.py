#! /usr/local/python3.6.5/bin//python3.6
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('/home/team55/notespace/zengbin')

# ======== Replace with your real import ======== #
from jddc.bm25 import run_prediction

if __name__ == '__main__':
    try:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    except IndexError:
        print('Two positional arguments are required: input_file_path, output_file_path')
        exit(1)

    if not os.path.isfile(input_file_path):
        print('Input file is not found: %s' % input_file_path)
        exit(1)

    output_dir = os.path.dirname(output_file_path)
    if not os.path.isdir(output_dir):
        print('Output dir is not exist: %s' % output_dir)
        exit(1)

    # ======== Replace with your real code ======== #
    run_prediction(input_file_path, output_file_path)

