# coding=utf-8

from gensim.summarization import bm25
from zb.tools.file_tools import read_file

from .config import conf, logger
from .utils import tokenize_with_jieba


class BM25Model:
    def __init__(self, fresh=False):
        self.fresh = fresh
        self.model = None
        self.average_idf = None

    @staticmethod
    def initialize():
        """初始化操作：1)分词"""
        logger.info("分词参数：input_file - %s; output_file - %s, "
                    "stopwords_file - %s; fro_search - %s" % (
                        conf.file_questions, conf.file_questions_segs,
                        conf.file_stopwords, str(conf.for_search)
                    ))
        segs = tokenize_with_jieba(input_file=conf.file_questions,
                                   output_file=conf.file_questions_segs,
                                   stopwords_file=conf.file_stopwords,
                                   for_search=conf.for_search)
        return segs

    def train(self):
        """训练BM25模型"""
        if self.fresh:
            logger.info("重新分词，创建模型")
            segs = self.initialize()
        else:
            logger.info("从%s读入现有分词结果，创建模型" % conf.file_questions_segs)
            segs = read_file(conf.file_questions_segs)
            segs = [eval(x) for x in segs]
        self.model = BM25Model = bm25.BM25(segs)
        self.average_idf = sum(map(lambda k: float(BM25Model.idf[k]),
                                   BM25Model.idf.keys())) / len(BM25Model.idf.keys())
        logger.info("BM25模型创建成功")

    def predict(self, input_file):
        """输入测试问题，查找最相似的问题"""
        logger.info("测试集：%s" % input_file)
        questions_segs = tokenize_with_jieba(input_file,
                                             for_search=conf.for_search,
                                             stopwords_file=conf.file_stopwords,
                                             )
        logger.info("输入文件 %s 分词完成" % input_file)
        model = self.model
        average_idf = self.average_idf

        results = []
        for q in questions_segs:
            scores = model.get_scores(q, average_idf)
            top_sims = sorted(enumerate(scores), key=lambda item: item[1],
                              reverse=True)[:conf.top]
            results.append(top_sims)
        return results



def run_prediction(input_file_path, output_file_path):
    model = BM25Model()
    model.train()

    logger.info('predict most similarity questions in %s' % input_file_path)
    sim_results = model.predict(input_file_path)

    logger.info("read reference answers from %s" % conf.file_answers)
    answers = read_file(conf.file_answers)

    logger.info("result will write to %s" % output_file_path)
    logger.info("max_len_answer is the longest answer of top %s" % str(conf.top))
    with open(output_file_path, 'w', encoding='utf-8') as file_result:
        for top_sims in sim_results:
            answer_list = []
            for j in range(0, len(top_sims)):
                answer_index = top_sims[j][0]
                answer = answers[answer_index]
                answer_list.append((answer, len(answer)))
            max_len_answer = sorted(answer_list, key=lambda x: x[1], reverse=True)[0][0]
            file_result.write(max_len_answer+'\n')


