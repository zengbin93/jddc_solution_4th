# coding: utf-8
"""
实用对象
====================================================================
"""

from jddc.utils import n_grams, jieba_tokenize


class Sentence(object):
    def __init__(self, sentence):
        self.origin = sentence

    def cut(self, for_search=False):
        return jieba_tokenize(self.origin, for_search)

    def n_grams_feature(self, n=3):
        return n_grams(self.origin, n)

    def get_tokens(self, method='tri-gram'):
        """根据method获取sentence的分词结果"""
        if method == "tri-gram":
            tokens = self.n_grams_feature(3)
        elif method == 'bi-gram':
            tokens = self.n_grams_feature(2)
        elif method == 'chars':
            tokens = self.n_grams_feature(1)
        elif method == "jieba_cut":
            tokens = self.cut()
        elif method == "jieba_cut_for_search":
            tokens = self.cut(for_search=True)
        else:
            raise ValueError
        return tokens


class Session(object):
    """会话对象：用户与客服的所有对话、相关产品等"""

    def __init__(self, sess_info):
        self.sess_info = sess_info
        self.is_repeat = sess_info['repeat'][0] if sess_info['repeat'][0] else '0'
        self.is_transfer = sess_info['transfer'][0] if sess_info['transfer'][0] else '0'
        self.user_id = sess_info['user_id']
        self.session_id = sess_info['session_id']

        # 固定参数配置
        self.qa_sep = '<s>'  # 不同人说话之间的分割符
        self.sentence_sep = " "  # 同一个人连续说多句话的分隔符
        self.q_sep = "<q>"  # 用户问题之间的分割符

    def sentence_pre(self, sentence):
        """sentence预处理

        :param sentence: 需要预处理的句子
        :type sentence: str
        """
        pass

    @property
    def data_quality(self):
        """数据质量检验"""
        qas = self.qas_merged
        # 对话轮数在 3~20之间
        if (qas is None) or (len(qas)/2 > 20) or (len(qas)/2 < 3):
            return False

        # 前3轮对话文本长度不要太长，也不能太短
        q, a = self.qaqaq_a
        if q == "" or a == "":
            return False

        if (len(q) > 500) or (len(q) < 30) \
                or (len(a) > 300) or (len(a) < 3):
            return False

        return True

    @property
    def raw_text(self):
        try:
            return self.sess_info['lines']
        except KeyError:
            return None

    @property
    def qas_merged(self):
        try:
            return self.sess_info['qas_merged']
        except KeyError:
            return None

    @property
    def qas(self):
        qas = self.sess_info['qas_merged']
        if qas[0][0] == "1":
            qas = qas[1:]
        if qas[-1][0] == "0":
            qas = qas[:-1]
        assert len(qas) % 2 == 0
        return qas

    @property
    def qaqaq_a(self):
        """构造单轮数据集"""
        qas = self.sess_info['qas_merged']

        if len(qas) < 6:
            return "", ""
        if qas[-1][0] == "0":
            qas = qas[:-1]
        if qas[0][0] == "1":
            qas = qas[1:]

        sep = self.qa_sep
        if len(qas) < 6:
            qaqaq = sep.join([x[1] for x in qas[:-1]])
            a = qas[-1][1]
        else:
            qaqaq = sep.join([x[1] for x in qas[:5]])
            a = qas[5][1]
        return qaqaq, a

    def multi_qa(self, merge_q=False):
        """构造多轮数据集"""
        multi = []
        qas = self.qas
        old_q = []
        for turn, i in enumerate(range(0, len(qas), 2), 1):
            old_q.append(qas[i][1].replace("\t", self.sentence_sep))
            a = qas[i + 1][1].replace("\t", self.sentence_sep)
            # session_id, 轮数, 累计Q, 当前轮次的回答
            if merge_q:
                row = [self.session_id, turn, "<q>".join(old_q), a]
            else:
                row = [self.session_id, turn, old_q[-1], a]
            multi.append(row)
        return multi

    def __repr__(self):
        return "<Session Object of %s>" % self.session_id


