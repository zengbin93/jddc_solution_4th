# coding=utf-8

import requests
import jieba
from thulac import thulac
import traceback
from retrying import retry
from zb.tools.file_tools import read_file


class ApiSeg(object):
    """京东API分词器"""
    def __init__(self, file_stopwords, api_token, inner):
        self.stopwords = read_file(file_stopwords)
        self.api_token = api_token
        self.inner = inner

    @staticmethod
    def _line_prepare(line):
        """预处理line，将特殊字符替换，清除空格"""
        str_tf = dict()
        str_tf["#E-s[数字x]"] = "微笑"
        str_tf["&nbsp;"] = ""
        str_tf["[数字x]%"] = "比例"
        str_tf["[金额x]%"] = "比例"
        str_tf["%"] = ""
        str_tf["#"] = ""
        str_tf["\t"] = ""
        str_tf[" "] = ""
        for k, v in str_tf.items():
            line = line.replace(k, v)
        return line

    @retry(stop_max_attempt_number=6)
    def cut(self, sentence, stopwords=True):
        # token = '9fb9785b4ea044e5871a8cbdae354e03'
        token = self.api_token
        sentence = self._line_prepare(sentence)
        # inner = True
        inner = self.inner
        headers = {
            "User-Agent": 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 '
                          'Mobile Safari/537.36',
        }
        url_outer = "http://jdialog-lexeme.jd.com/lexeme?token={token}&text={text}"
        url_inner = "http://jdialog-lexeme-stage.jd.com/lexeme?token={token}&text={text}"
        if inner:
            url = url_inner.format(token=token, text=sentence)
        else:
            url = url_outer.format(token=token, text=sentence)
        res = requests.get(url, headers=headers).json()
        assert res['status'] == 0
        tokens = [x['word'] for x in res['tokenizedText']]
        results = []
        for seg in tokens:
            if seg in self.stopwords and stopwords:
                continue  # 去除停用词
            results.append(seg)
        return results

    def cut_for_search(self, sentence, stopwords=True):
        try:
            results = self.cut(sentence, stopwords)
        except:
            results = []
            traceback.print_exc()
        return results


class JiebaSeg(object):
    """jieba分词器"""
    def __init__(self, file_stopwords):
        self.stopwords = read_file(file_stopwords)

    def cut(self, sentence, stopwords=True):
        seg_list = jieba.cut(sentence)  # 切词

        results = []
        for seg in seg_list:
            if seg in self.stopwords and stopwords:
                continue  # 去除停用词
            results.append(seg)
        return results

    def cut_for_search(self, sentence, stopwords=True):
        seg_list = jieba.cut_for_search(sentence)

        results = []
        for seg in seg_list:
            if seg in self.stopwords and stopwords:
                continue
            results.append(seg)
        return results


class LacSeg(object):
    def __init__(self, clear_sw=True):
        self.name = "thulac"
        self.tl = thulac(filt=clear_sw)

    def cut(self, sentence):
        lac_tokens = self.tl.cut(sentence)
        return [x[0] for x in lac_tokens]
    
    def cut_for_search(self, sentence):
        return self.cut(sentence)
        


class Sentence(object):
    def __init__(self, sentence, seg, num=0):
        self.id = num
        self.origin_sentence = sentence
        self.cuted_sentence = self.cut(seg)

    # 对句子分词
    def cut(self, seg):
        return seg.cut(self.origin_sentence)

    # 获取切词后的词列表
    def get_cuted_sentence(self):
        return self.cuted_sentence

    # 获取原句子
    def sentence_origin(self):
        return self.origin_sentence
