import re
from tqdm import tqdm
from pprint import pprint
import random
from gensim.summarization import bm25

from jddc.config import BM25Config
from jddc.utils import write_file, read_file
from jddc.utils import save_to_pkl, read_from_pkl
from jddc.utils import create_logger
from jddc.utils import n_grams, jieba_tokenize
from jddc.datasets import read_test_questions02

conf = BM25Config()
logger = create_logger(name='bm25', log_file=conf.log_file, cmd=conf.cmd_log)

random.seed(conf.random_num)


# 模型创建/加载
# --------------------------------------------------------------------
def create_bm25_model(questions, answers):
    """从头创建模型"""
    questions_tokens = []
    for q in tqdm(questions, desc="cut"):
        q_tokens = n_grams(q, conf.n)
        questions_tokens.append(q_tokens)

    model = bm25.BM25(questions_tokens)
    average_idf = sum(float(val) for val in
                      model.idf.values()) / len(model.idf)
    data = [model, answers, average_idf]
    save_to_pkl(file=conf.pkl_bm25, data=data)
    return model, answers, average_idf


def load_bm25_model():
    return read_from_pkl(conf.pkl_bm25)


def get_bm25_scores(model, document, indexes, average_idf):
    """Computes and returns BM25 scores of given `document` in relation to
    every item in corpus.

    Parameters
    ----------
    model : bm25 Model
    document : list of str
        Document to be scored.
    indexes :
    average_idf : float
        Average idf in corpus.

    Returns
    -------
    list of float
        BM25 scores.
    """
    scores = []
    for index in indexes:
        score = model.get_score(document, index, average_idf)
        scores.append(score)
    return scores


def run_prediction(input_file, output_file):
    # 加载模型
    print(str(conf))
    model, answers, average_idf = load_bm25_model()
    print("load model from %s success." % conf.pkl_bm25)

    # 处理输入的测试数据集
    # ********************************************************************
    test_set = read_test_questions02(input_file)

    # 开始预测
    # ********************************************************************
    print("start predicting ...")
    predicted_answers = []
    for q in tqdm(test_set, ncols=100, desc="run_prediction"):
        print("=" * 88)
        print("question:", q)
        q_tokens = jieba_tokenize(q)
        indexes = list(range(model.corpus_size))
        scores = get_bm25_scores(model=model, document=q_tokens,
                                 indexes=indexes, average_idf=average_idf)
        top_sims = sorted(enumerate(scores), key=lambda item: item[1],
                          reverse=True)[:conf.top]
        candidates = []
        for x in top_sims:
            candidates.append(answers[x[0]].replace("\t", " "))
        # 选最长
        candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
        pprint(candidates)
        predicted_answers.append(candidates[0])
    write_file(file=output_file, content=predicted_answers, mode='w', encoding='utf-8')

