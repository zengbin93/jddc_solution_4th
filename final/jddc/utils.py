# coding: utf-8
from threading import Lock
import os
import pickle
import codecs
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import jieba

lock = Lock()


class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def jieba_tokenize(text, for_search=False):
    if for_search:
        tokens = list(jieba.cut_for_search(text))
    else:
        tokens = list(jieba.cut(text))
    return tokens


def create_logger(log_file, name='logger', cmd=True):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # set format
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # cmd handler
    if cmd:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# pickle文件操作
# --------------------------------------------------------------------
def save_to_pkl(file, data, protocol=None):
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def read_from_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


# 文件夹操作
# --------------------------------------------------------------------
def insure_folder_exists(folder):
    """确保文件夹存在"""
    if not os.path.exists(folder):
        os.mkdir(folder)


# txt文件操作
# --------------------------------------------------------------------
def write_file(file, content=None, mode="a", encoding='utf-8'):
    lock.acquire(blocking=True, timeout=30)
    with open(file, mode, encoding=encoding) as f:
        if isinstance(content, str):
            content += "\n"
            f.write(content)
        elif isinstance(content, list):
            content = [i.strip("\n") + '\n' for i in content]
            f.writelines(content)
        elif content is None:
            pass
        else:
            raise ValueError("If content is not None, it must be list or str!")
    lock.release()


def read_file(file, mode="r", encoding='utf-8'):
    lock.acquire(blocking=True, timeout=30)
    with open(file, mode, encoding=encoding) as f:
        lines = f.readlines()
        lines = [line.strip("\n") for line in lines]
    lock.release()
    if len(lines) > 0:
        return lines
    else:
        raise ValueError("file is empty!")


def empty_file(file, mode="w", encoding='utf-8'):
    """empty file"""
    with open(file, mode, encoding=encoding) as f:
        f.truncate()


# n-gram特征提取
# --------------------------------------------------------------------
def n_grams(text, n=3):
    """基于字符的n-gram切词

    :param text: str
        文本
    :param n: int
        n-grams参数
    :return: list
        提取的n元特征
    """
    chars = list(text)
    grams = []
    if len(chars) < n:
        grams.append("".join(chars))
    else:
        for i in range(len(chars) - n + 1):
            gram = "".join(chars[i:i + n])
            grams.append(gram)
    return grams


# bleu得分计算
# --------------------------------------------------------------------
def bleu(answerFilePath, standardAnswerFilePath):
    with codecs.open(answerFilePath, 'r', "utf-8") as rf_answer:
        with codecs.open(standardAnswerFilePath, 'r', "utf-8") as rf_standardAnswer:
            score = []
            answerLines = rf_answer.readlines()
            standardAnswerLines = rf_standardAnswer.readlines()
            chencherry = SmoothingFunction()
            for i in range(len(answerLines)):
                candidate = list(answerLines[i].strip())
                eachScore = 0
                # 10个标准答案
                for j in range(10):
                    reference = []
                    standardAnswerLine = standardAnswerLines[i * 11 + j].strip().split('\t')
                    reference.append(list(standardAnswerLine[0].strip()))
                    standardScore = standardAnswerLine[1]
                    # bleu
                    bleuScore = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                                              smoothing_function=chencherry.method1)
                    # 加权平均
                    eachScore = bleuScore * float(standardScore) + eachScore
                print("answer %i 得分：" % i, eachScore / 10)
                score.append(eachScore / 10)

            rf_answer.close()
            rf_standardAnswer.close()
            # 50个QAQAQ + A单轮测试得分
            print("50个单轮QA测评得分：", sum(score[:50])/50)
            # 最终得分
            scoreFinal = sum(score) / float(len(answerLines))
            # 最终得分精确到小数点后6位
            precisionScore = round(scoreFinal, 6)
            return precisionScore


def cal_bleu_score(candidate, reference):
    """计算bleu得分

    :param candidate: str
        候选答案
    :param reference: str
        参考答案
    :return: score
    """
    reference = list(reference.strip())
    candidate = list(candidate.strip())
    chen_cherry = SmoothingFunction()
    score = sentence_bleu(references=[reference],
                          hypothesis=candidate,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=chen_cherry.method1)
    return score



