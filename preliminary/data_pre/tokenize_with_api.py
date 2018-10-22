# coding: utf-8

"""
使用京东API进行分词
=============================================================================
"""

import requests
import grequests
from zb.tools.logger import create_logger

from jddc_utils import file_op

# 配置参数
# ------------------------------------------------------------------------------

from data_pre.config import BaseConf
conf = BaseConf()

logger = create_logger(conf.log_file, name="tokenize", cmd=conf.cmd_log)

def get_text_tokenize(text):
    """调用京东的分词器API"""
    token = conf.api_token
    inner = conf.inner
    headers = {
        "User-Agent": 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 '
                      'Mobile Safari/537.36',
    }
    url_outer = "http://jdialog-lexeme.jd.com/lexeme?token={token}&text={text}"
    url_inner = "http://jdialog-lexeme-stage.jd.com/lexeme?token={token}&text={text}"

    if inner:
        url = url_inner.format(token=token, text=text)
    else:
        url = url_outer.format(token=token, text=text)

    return requests.get(url, headers=headers).json()


# 测试京东api对异常字符的处理方法
# ------------------------------------------------------------------------------
def test_api():
    texts = [
        "#E-s[数字x]",
        "&nbsp;[数字x]",
        "请问还有什么小秘书可以帮您的呢 ~   (*^__^*)#E-s[数字x]",
        " [站点x]贝兜 宝宝零食肉松 什锦[姓名x]肉绒罐装 [数字x]g([数字x]小包)",
        "[站点x] 空调类京东安装收费标准://help.jd.com/user/issue/[数字x]-[链接x]",
        "如果您遗忘/未设置密码 还麻烦亲拨打运营商人工客服帮您进行重置下呢，"
        "小妹可以更快的帮您解决您的问题哦#E-s[数字x]#E-s[数字x]",
        "店铺ID:&nbsp;[数字x]",
        "订单号:&nbsp;[数字x]",
        "降幅[数字x]%",
        "订单完成 返  [金额x]%",
        "我看到扣点是[数字x]%，但是我想了解一下，我们还需要[姓名x]么?",
    ]

    for text in texts:
        text = text.replace("#E-s[数字x]", "微笑")
        text = text.replace("#E-j[数字x]", "愤怒")
        text = text.replace("&nbsp;", " ")
        print(text, get_text_tokenize(text))


# 测试京东api对异常字符的处理方法
# ------------------------------------------------------------------------------

def line_pre(line):
    """行数据预处理"""
    line_ = line.split("\t")
    if len(line_) > 7:
        line_pred = line_[:6]
        text = " ".join(line_[6:])
        line_pred.append(text)
        line = '\t'.join(line_pred)

    str_tf = {
        "#E-s[数字x]": "微笑",
        "#E-j[数字x]": "愤怒",
        "&nbsp;": " ",
        "[数字x]%": "比例",
        "[金额x]%": "比例",
        "%": " ",
        "#": " ",
    }
    for k, v in str_tf.items():
        line = line.replace(k, v)
    return line

# 异步调用实现
# ------------------------------------------------------------------------------

def async_batch_tokenize(chat_batch):
    """批量处理"""
    inner = conf.inner
    url_outer = "http://jdialog-lexeme.jd.com/lexeme?token={token}&text={text}"
    url_inner = "http://jdialog-lexeme-stage.jd.com/lexeme?token={token}&text={text}"
    if inner:
        base_url = url_inner
    else:
        base_url = url_outer
    urls = map(lambda x: base_url.format(token=conf.api_token,
                                         text=x.strip('\r\n').split('\t')[6]), chat_batch)

    def exception_handler(request, e):
        logger.error(request)
        logger.exception(e)

    tasks = [grequests.get(url, timeout=3) for url in urls]
    results = grequests.map(tasks, size=20, exception_handler=exception_handler)

    logger.info("finish batch api request, start collect res ...")
    chat_tokenize = []
    for i, res in enumerate(results):
        line = chat_batch[i]
        try:
            res_json = res.json()
            if res_json['status'] == 0:
                line += "\t" + str(res_json['tokenizedText'])
            else:
                line += "\t" + "fail_tokenize"
        except Exception as e:
            line += "\t" + "fail_tokenize"
            logger.exception(e)
        chat_tokenize.append(line)

    # save
    file_chat_tokenize = conf.file_chat_tokenize
    logger.info("saving chat batch tokenize to %s" % file_chat_tokenize)
    file_op.write_lines(file_chat_tokenize, chat_tokenize)
    logger.info("save success!")

def async_tokenize(batch=100, empty=False):
    """batch不能设置过大！越大，返回错误的次数越多，而且速度也变慢了

    默认是断点恢复模式，即从上回的结束位置继续；
    设置 empty 为 True，将从头开始。
    """
    if empty:
        logger.info("refresh tokenize mode!")
        logger.info("empty chat tokenize %s" % conf.file_chat_tokenize)
        file_op.empty_file(conf.file_chat_tokenize)
        start = 0
    else:
        chat_tokenize = file_op.read_lines(conf.file_chat_tokenize)
        start = len(chat_tokenize)
        logger.info("breakpoint resume mode, start is %s" % str(start))

    file_chat = conf.file_chat_pred
    chat = file_op.read_lines(file_chat)
    logger.info("read %s success!" % file_chat)
    chat[0] = chat[0].replace("\ufeff", '')
    logger.info("tokenize from %i" % start)
    chat = chat[start:] 
    batch_nums = int(len(chat) / batch) + 1
    for i in range(batch_nums):
        chat_batch = chat[i*batch:(i+1)*batch]
        logger.info("current batch start is %s" % str(i*batch))
        async_batch_tokenize(chat_batch)

# 修复第一遍调用API处理失败的text
# ------------------------------------------------------------------------------

def inspect_tokenize(line):
    """检查某一行的分词结果是否正确返回"""
    if line.split("\t")[7] == 'fail_tokenize':
        return False
    else:
        tokens = eval(line.split("\t")[7])
        text = ''.join([x['word'] for x in tokens])
        if line.split('\t')[6].replace(" ", "") == text:
            return True
        else:
            return False

def count_fail():
    """统计调用api失败的数量"""
    logger.info("count api invoke fail numbers.")
    file_chat_tokenize = conf.file_chat_tokenize
    chat_tokenize = file_op.read_lines(file_chat_tokenize)
    logger.info("read %s success!" % chat_tokenize)
    nums = {
        "fail": 0,
        "success": 0
    }

    for line in chat_tokenize:
        try:
            if inspect_tokenize(line):
                nums["success"] += 1
            else:
                nums["fail"] += 1
        except:
            nums["fail"] += 1
    logger.info("tokenize success nums: %i, fail nums: %i" %
                (nums["success"], nums["fail"]))
    return nums


def tokenize_modify():
    """对部分返回失败的句子重新分词"""
    logger.info("modify tokenize results!")
    file_chat_tokenize = conf.file_chat_tokenize
    chat_tokenize = file_op.read_lines(file_chat_tokenize)
    logger.info("read %s success!" % chat_tokenize)
    for i in range(len(chat_tokenize)):
        line = chat_tokenize[i]
        try:
            if not inspect_tokenize(line):
                line = "\t".join(line.split('\t')[:-1])
                # 处理特殊字符
                line = line_pre(line)
                text = line.split('\t')[6]
                logger.info("current line: %i, text: %s" % (i, text))
                res = get_text_tokenize(text)
                text_tokens = res.get('tokenizedText', "fail_tokenize")
                line += str(text_tokens)
                chat_tokenize[i] = line
        except Exception as e:
            logger.exception(e)
            print(line)
    file_op.write_lines(file_chat_tokenize, chat_tokenize, mode='w')
    logger.info("write %s success!" % chat_tokenize)
