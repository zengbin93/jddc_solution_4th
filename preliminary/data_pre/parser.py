# -*- coding: UTF-8 -*-

"""
1. 根据session_id划分对话，将每一个对话所有行归集，并统计q、a的数量，
结果文件 chat_parsed.txt

2、读入chat_parsed.txt，解析每一个会话，合并连续q、a，提取订单信息等，
结果文件 session_parsed.txt
================================================================================
"""

import re
from zb.tools.logger import create_logger

from data_pre.config import BaseConf
from jddc_utils import file_op

base_conf = BaseConf()

logger = create_logger(base_conf.log_file, name='pre', cmd=True)
logger.info("Logger create success, log file is %s" % base_conf.log_file)

def _init_res_file():
    file_qaqaq = base_conf.file_qaqaq
    file_a = base_conf.file_a
    file_op.empty_file(file_qaqaq)
    file_op.empty_file(file_a)
    return file_qaqaq, file_a

# ------------------------------------------------------------------------------

def _update_nums(sess_info, line_cols):
    """统计每一个session中用户和客服的说话次数"""
    if line_cols['waiter_send'] == '0':
        sess_info['q_nums'] += 1
    elif line_cols['waiter_send'] == '1':
        sess_info['a_nums'] += 1
    else:
        logger.info("waiter_send value must be 0 or 1: %s" % '\t'.join(line_cols))
    return sess_info

def chat_parse():
    """逐行解析chat数据，按照session进行统计，归集"""
    file_chat = base_conf.file_chat
    logger.info('reading chat from %s' % file_chat)
    lines = file_op.read_lines(file_chat)
    chat_parsed = []

    # 初始化 session info
    sess_info = {
        "session_id": lines[0].split('\t')[0],
        "q_nums": 0,
        "a_nums": 0,
        "lines": []
    }

    for line in lines:
        line = line.strip('\t').replace("\t", '|')
        try:
            cols = line.split("|")
            line_cols = {
                "id": cols[0],
                "user": cols[1],
                "waiter_send": cols[2],
                "transfer": cols[3],
                "repeat": cols[4],
                "sku": cols[5],
                "content": "|".join(cols[6:])
            }
            # assert len(cols) == 7, "总共有七个字段，当前行有%i个字段" % len(cols)
            if sess_info['session_id'] == line_cols['id']:
                sess_info = _update_nums(sess_info, line_cols)
                sess_info['lines'].append(line)
            else:
                chat_parsed.append(sess_info)
                sess_info = {
                    "session_id": line_cols['id'],
                    "q_nums": 0,
                    "a_nums": 0,
                    "lines": [line]
                }
                sess_info = _update_nums(sess_info, line_cols)
        except Exception as e:
            logger.error('line error: %s' % line)
            logger.exception(e)
    file_op.write_lines(base_conf.file_chat_parsed, chat_parsed)
    logger.info("chat parse result saved in %s" % base_conf.file_chat_parsed)
    return chat_parsed

# ------------------------------------------------------------------------------

def _parse_session(sess_info):
    """解析单个session，获取order_id等信息

    返回结果：
        {
        "session_id": 会话id,
        "user_id": user_id,
        "order_id": order_id,
        "sku": 商品品类,
        "transfer": 是否转移,
        "repeat": 是否重复,
        "lines": 原始数据行,
        "qas_merged": 合并之后的对话记录
        }
    """
    lines = sess_info['lines']
    user_id = lines[0].split("\t")[1]
    transfer = list(set([line.split("\t")[3] for line in lines
                         if line.split("\t")[3] != '']))
    repeat = list(set([line.split("\t")[4] for line in lines
                       if line.split("\t")[4] != '']))
    sku = list(set([line.split("\t")[5] for line in lines
                    if line.split("\t")[5] != '']))

    # 提取订单号
    contents = "\t".join([line.split("\t")[6] for line in lines])
    pat_oid = re.compile(r'(ORDERID_\d{8})')
    order_id = list(set(pat_oid.findall(contents)))

    # 合并q/a
    qas = [(line.split("\t")[2], line.split("\t")[6]) for line in lines]
    qas_merged = []
    current_sender = qas[0][0]
    content = qas[0][1]
    for qa in qas[1:]:
        if current_sender == qa[0]:
            content += "\t" + qa[1]
            # 尾行处理
            if qa == qas[-1]:
                qa_ = (current_sender, content)
                qas_merged.append(qa_)
        else:
            qa_ = (current_sender, content)
            qas_merged.append(qa_)
            current_sender = qa[0]
            content = qa[1]
            # 尾行处理
            if qa == qas[-1]:
                qa_ = (current_sender, content)
                qas_merged.append(qa_)

    return {
        "session_id": sess_info['session_id'],
        "user_id": user_id,
        "order_id": order_id,
        "sku": sku,
        "transfer": transfer,
        "repeat": repeat,
        "lines": lines,
        "qas_merged": qas_merged
    }


def chat_session_parse():
    """输入chat_parse()返回的结果，将连续的q、a进行合并，并标记顺序"""
    logger.info("reading chat parsed from %s" % base_conf.file_chat_parsed)
    chat_parsed = file_op.read_lines(base_conf.file_chat_parsed)
    chat_parsed = [eval(x) for x in chat_parsed]
    session_parsed = []
    for sess_info in chat_parsed:
        try:
            sess_parsed = _parse_session(sess_info)
            session_parsed.append(sess_parsed)
        except Exception as e:
            logger.error("sess info parse error, sess_id: %s" % sess_info['session_id'])
            logger.exception(e)
    file_session_parsed = base_conf.file_session_parsed
    logger.info("save session parse result to %s" % file_session_parsed)
    file_op.write_lines(file_session_parsed, session_parsed)
    logger.info("save success!")

# save to json test
# ------------------------------------------------------------------------------

def json_wr():
    import json
    file_session_parsed = base_conf.file_session_parsed
    json_session_parsed = base_conf.json_session_parsed
    session_parsed = file_op.read_lines(file_session_parsed)
    session_parsed = [eval(x) for x in session_parsed]

    sp_dict = dict()
    for x in session_parsed:
        del x['lines']
        sp_dict[x['session_id']] = x

    json.dump(sp_dict, open(json_session_parsed, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    # xx = json.load(open(json_session_parsed, 'r', encoding='utf-8'))

