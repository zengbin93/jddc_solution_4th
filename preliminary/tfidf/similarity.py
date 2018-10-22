"""
基于TFIDF的句子相似性计算
====================================================================
"""
import os
import pickle
from tqdm import tqdm
from gensim import corpora, models, similarities
from .utils import Sentence
from collections import defaultdict


class SentenceSimilarity(object):
    def __init__(self, seg, model_path=None):
        self.seg = seg   # seg是分词器
        self.sentences = []
        self.model = None
        self.model_path = model_path
        self.texts = None
        self.dictionary = None
        self.corpus_simple = None
        self.corpus = None
        self.index = None
        if self.model_path is not None:
            self.pkl_file = os.path.join(self.model_path, "corpus_and_dictionary.pkl")
            self.file_model = os.path.join(self.model_path, "tfidf.model")
            self.file_index = os.path.join(self.model_path, 'index.index')

    def set_sentences(self, sentences):
        self.sentences = []

        for i in tqdm(range(0, len(sentences)), ncols=100):
            self.sentences.append(Sentence(sentences[i], self.seg, i))

    def get_cuted_sentences(self):
        """获取切过词的句子"""
        cuted_sentences = []

        for sentence in tqdm(self.sentences, ncols=100):
            cuted_sentences.append(sentence.get_cuted_sentence())

        return cuted_sentences

    # 构建其他复杂模型前需要的简单模型
    def simple_model(self, min_frequency=1):
        self.texts = self.get_cuted_sentences()

        # 删除低频词
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1

        self.texts = [[token for token in text if frequency[token] > min_frequency]
                      for text in self.texts]

        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def TfidfModel(self):
        """tfidf模型"""
        if os.path.exists(self.file_model):
            print("model is exists, prepare to load it.")
            with open(self.pkl_file, 'rb') as f:
                self.dictionary, self.corpus_simple = pickle.load(f)
            self.model = models.TfidfModel.load(self.file_model)
            self.index = similarities.MatrixSimilarity.load(self.file_index)
            print('load tfidf model successfully.')
        else:
            print("train tfidf model from the beginning")
            self.simple_model()
            # 转换模型
            self.model = models.TfidfModel(self.corpus_simple)
            self.corpus = self.model[self.corpus_simple]
            # 创建相似度矩阵
            self.index = similarities.MatrixSimilarity(self.corpus)
            print("create tfidf model success.")
            # 保存模型
            if self.model_path is not None:
                with open(self.pkl_file, 'wb') as f:
                    pickle.dump([self.dictionary, self.corpus_simple], f)
                self.model.save(self.file_model)
                self.index.save(self.file_index)

    def sentence2vec(self, sentence):
        sentence = Sentence(sentence, self.seg)
        vec_bow = self.dictionary.doc2bow(sentence.get_cuted_sentence())
        return self.model[vec_bow]

    def similarity(self, sentence, top=15):
        """求最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)
        sims = self.index[sentence_vec]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_15 = sim_sort[0:top]

        return top_15
