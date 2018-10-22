# coding: utf-8
"""
解决方案：Tfidf + Cosine
====================================================================
"""

from tqdm import tqdm
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint

from jddc.obj import Sentence
from jddc.utils import save_to_pkl, read_from_pkl
from jddc.utils import write_file
from jddc.datasets import read_test_questions01, read_test_questions02
from jddc.config import TfidfConfig


conf = TfidfConfig()


class TfidfSim(object):
    def __init__(self):
        self.questions = None
        self.answers = None
        self.model = None
        self.texts = None
        self.dictionary = None
        self.corpus_simple = None
        self.corpus = None
        self.index = None

    def set_data(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.texts = []
        for i in tqdm(range(0, len(questions)), ncols=100, desc='cut_questions'):
            s = Sentence(questions[i])
            self.texts.append(s.n_grams_feature(3))

    def simple_model(self, min_frequency=6):
        # 删除低频词
        print("clear low-frequency words, min_frequency is %s" % str(min_frequency))
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > min_frequency]
                      for text in self.texts]

        print("create dictionary of corpus")
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def tfidf_model(self):
        """tfidf模型"""
        self.simple_model()
        # 转换模型
        print("train TfidfModel")
        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]
        # 创建相似度矩阵
        print("create MatrixSimilarity")
        self.index = similarities.MatrixSimilarity(self.corpus)
        # self.index = similarities.Similarity(output_prefix=prefix, corpus=self.corpus,
        #                                      num_features=len(self.dictionary))

    def question2vec(self, question):
        if isinstance(question, str):
            sentence = Sentence(question)
            q_tokens = sentence.n_grams_feature(3)
        elif isinstance(question, list):
            q_tokens = question
        else:
            raise ValueError("param question must be str or list type.")
        vec_bow = self.dictionary.doc2bow(q_tokens)
        return self.model[vec_bow]

    def similarity(self, question, top=15):
        """求最相似的句子"""
        question_vec = self.question2vec(question)
        sims = self.index[question_vec]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_sims = sim_sort[0:top]

        results = [(x[1], self.questions[x[0]],
                    self.answers[x[0]]) for x in top_sims]
        return results

    def get_candidates(self, question, top=15):
        """获取候选答案"""
        results = self.similarity(question, top)
        return [x[2] for x in results]

    def clear(self):
        """得到模型后，清理对象"""
        self.corpus = None
        self.corpus_simple = None
        self.texts = None


# 模型创建 / 加载
# --------------------------------------------------------------------
def create_tfidf_ir_model(questions, answers):
    print("create tfidf_sim model, set data")
    ts = TfidfSim()
    ts.set_data(questions, answers)

    print("train tfidf_sim model")
    ts.tfidf_model()
    ts.clear()

    print("save model")
    save_to_pkl(file=conf.pkl_tfidf_ir, data=ts)


def load_tfidf_ir_model():
    return read_from_pkl(conf.pkl_tfidf_ir)


# 模型提交入口函数
# --------------------------------------------------------------------
def run_prediction(input_file, output_file):
    # 加载模型
    print(str(conf))
    model = load_tfidf_ir_model()
    print("load model from %s success." % conf.pkl_tfidf_ir)

    # 读入测试数据集
    # ********************************************************************
    test_set = read_test_questions02(input_file)

    # 开始预测
    # ********************************************************************
    print("start predicting ...")
    predicted_answers = []
    for i, q in tqdm(enumerate(test_set), ncols=100, desc="run_prediction"):
        results = model.similarity(q, top=conf.top)
        print(i, "**"*68)
        print("questions: ", q)
        pprint(results)
        results = sorted(results, key=lambda x: len(x[2]), reverse=True)
        predicted_answers.append(results[0][2])
    write_file(file=output_file, content=predicted_answers, mode='w', encoding='utf-8')



