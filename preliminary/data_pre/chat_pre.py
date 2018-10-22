# -*- coding: UTF-8 -*-

import re
from zb.tools.logger import create_logger
from jddc_utils import file_op

# 配置参数
# ------------------------------------------------------------------
from data_pre.config import BaseConf
base_conf = BaseConf()

conf = {
    "chat": base_conf.file_chat,
    "chat_pred": base_conf.file_chat_pred,
    "log_file": base_conf.log_file,
    "cmd_log": base_conf.cmd_log
}

logger = create_logger(conf['log_file'], name='chat_pre', cmd=conf['cmd_log'])

# ------------------------------------------------------------------

def transform_text(text):
    """特殊字符转换"""
    str_tf = {
        "#E-s[数字x]": "微笑",
        "#E-j[数字x]": "愤怒",
        "&nbsp;": " ",
        "[数字x]%": "比例",
        "[金额x]%": "比例",
        "%": " ",
        "#": " ",
        "&": " ",
    }
    for k, v in str_tf.items():
        text = text.replace(k, v)
    return text

def merge_78():
    """合并多余的字段"""
    lines = file_op.read_lines(conf['chat'])
    logger.info("read chat.txt success!")
    for i in range(len(lines)):
        line = lines[i]
        line_ = line.strip("\r\n").split('\t')
        if len(line_) > 7:
            line_pred = line_[:6]
            text = " ".join(line_[6:])
            line_pred.append(text)
            lines[i] = '\t'.join(line_pred)
            # print('\t'.join(line_pred))
    file_op.write_lines(conf['chat_pred'], lines)
    logger.info("write results to  %s success!" % conf['chat_pred'])


# ------------------------------------------------------------------

def get_bracket_words():
    lines = file_op.read_lines(conf['chat_pred'])
    logger.info("read %s success!" % conf['chat_pred'])
    bracket_pat = re.compile("\[(.*?)\]")
    bracket_values = []
    for line in lines:
        values = bracket_pat.findall(line)
        bracket_values.extend(values)
    bracket_values = list(set(bracket_values))
    # order_ids = [value for value in bracket_values if "ORDERID_" in value]
    values = [value for value in bracket_values
              if "ORDERID_" not in value and "USERID_" not in value]
    file_op.write_lines("temp.txt", values)


