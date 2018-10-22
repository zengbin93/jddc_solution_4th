# !/usr/bin/python3
# coding: utf-8

from data_pre.tokenize_with_api import async_tokenize
from data_pre.tokenize_with_api import count_fail
from data_pre.tokenize_with_api import tokenize_modify


def run_pipeline():
    # 借助京东的API对原始数据进行分词，结果添加到最后一列，结果文件：chat_tokenize.txt
    async_tokenize(empty=True)
    # tokenize_modify()


if __name__ == "__main__":
    run_pipeline()
