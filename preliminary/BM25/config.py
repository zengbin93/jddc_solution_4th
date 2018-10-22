# coding: utf-8

import os
from zb.tools.logger import create_logger
from zb.tools.dir_tools import insure_folder_exists


class Config(object):
    # base path
    base_path = "/submitwork/data"
    file_stopwords = os.path.join(base_path, 'stopwords.txt')

    # data path
    data_path = os.path.join(base_path, 'prepared')
    # 在这里切换QA数据集，不同数据集对得分的影响较大
    file_questions = os.path.join(data_path, 'questions.txt')
    file_answers = os.path.join(data_path, 'answers.txt')

    # conf for results
    res_path = os.path.join(base_path, 'BM25')
    insure_folder_exists(res_path)
    file_questions_segs = os.path.join(res_path, 'questions_segs.txt')
    log_file = os.path.join(res_path, 'bm25_implement.log')

    # other
    top = 5
    cmd_log = True
    for_search = True
    api_token = '9fb9785b4ea044e5871a8cbdae354e03'
    inner = False

# ------------------------------------------------------------------------------------------------------

conf = Config()
logger = create_logger(name='bm25', log_file=conf.log_file, cmd=conf.cmd_log)


