# -*- coding: UTF-8 -*-
import os

from zb.tools.dir_tools import insure_folder_exists

class BaseConf(object):
    # raw data
    raw_path = "C:\ZB\git_repo\JDDC\data\preliminaryData"
    # raw_path = "/mnt/dataset/preliminaryData"
    file_chat = os.path.join(raw_path, 'chat.txt')
    file_ware = os.path.join(raw_path, 'ware.txt')
    file_order = os.path.join(raw_path, 'order.txt')
    file_user = os.path.join(raw_path, 'user.txt')

    base_path = "C:\ZB\git_repo\JDDC\data"
    # base_path = "/submitwork/data"
    log_file = os.path.join(base_path, 'data_preprocessing.log')

    # conf for results
    res_path = os.path.join(base_path, 'prepared')
    insure_folder_exists(res_path)
    file_chat_parsed = os.path.join(res_path, 'chat_parsed.txt')
    file_chat_pred = os.path.join(res_path, 'chat_pred.txt')
    file_chat_tokenize = os.path.join(res_path, 'chat_tokenize.txt')
    file_session_parsed = os.path.join(res_path, 'session_parsed.txt')
    file_qaqaq = os.path.join(res_path, 'questions.txt')
    file_a = os.path.join(res_path, 'answers.txt')

    # json file
    json_session_parsed = os.path.join(res_path, 'session_parsed.json')

    # other
    chat_sep = "<s>"
    cmd_log = True
    api_token = '9fb9785b4ea044e5871a8cbdae354e03'
    inner = False



