# coding: utf-8

import os


class BaseConf:
    # 是否刷新模型，
    refresh_model = True
    # base_path = r"C:\ZB\git_repo\JDDC\data"
    base_path = "/submitwork/data"

    data_path = os.path.join(base_path, 'prepared')
    file_stopwords = os.path.join(data_path, "stopwords.txt")
    file_questions = os.path.join(data_path, 'questions.txt')
    file_answers = os.path.join(data_path, 'answers.txt')

    model_path = os.path.join(base_path, "TFIDF")

    log_file = os.path.join(model_path, 'tfidf.log')
    cmd_log = True
    # 从相似度top中选择最长的答案
    top = 15

    # 分词器选择 - lac/jieba/api
    seg_name = 'lac'

    # 使用京东api
    use_api = False
    api_token = '****************************'
    inner = True

