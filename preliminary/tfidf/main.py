# coding=utf-8
from zb.tools.logger import create_logger
from zb.tools.file_tools import read_file
import os

from .utils import JiebaSeg, ApiSeg, LacSeg
from .similarity import SentenceSimilarity
from .config import BaseConf

conf = BaseConf()

logger = create_logger(conf.log_file, name='tfidf', cmd=conf.cmd_log)

if conf.refresh_model:
    logger.info("refresh model, prepare to delete old model file.")
    pkl_file = os.path.join(conf.model_path, "corpus_and_dictionary.pkl")
    file_model = os.path.join(conf.model_path, "tfidf.model")
    file_index1 = os.path.join(conf.model_path, 'index.index')
    file_index2 = os.path.join(conf.model_path, 'index.index.index.npy')
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
    if os.path.exists(file_model):
        os.remove(file_model)
    if os.path.exists(file_index1):
        os.remove(file_index1)
    if os.path.exists(file_index2):
        os.remove(file_index2)
    logger.info("old model file deleted.")


def run_prediction(input_file_path, output_file_path):

    logger.info("run prediction, params: top - %s, "
                "file_questions - %s, file_answers - %s" % (
                    str(conf.top), conf.file_questions,
                    conf.file_answers
                ))

    # 读入训练集
    logger.info("read answers from: %s" % conf.file_answers)
    answers = read_file(conf.file_answers)

    # 分词工具
    logger.info("seg_name is : %s" % conf.seg_name)
    if conf.seg_name == 'api':
        seg = ApiSeg(conf.file_stopwords, api_token=conf.api_token,
                     inner=conf.inner)
    elif conf.seg_name == 'jieba':
        # 基于jieba分词，并去除停用词
        seg = JiebaSeg(conf.file_stopwords)
    elif conf.seg_name == 'lac':
        seg = LacSeg(clear_sw=True)
    else:
        raise ValueError("conf.seg_name值错误，可选值['jieba', 'lac', 'api']")

    # 训练模型
    ss = SentenceSimilarity(seg, model_path=conf.model_path)
    if not os.path.exists(os.path.join(conf.model_path, 'tfidf.model')):
        logger.info("refresh model, read questions from: %s" % conf.file_questions)
        questions = read_file(conf.file_questions)
        ss.set_sentences(questions)
    logger.info("starting train model.")
    ss.TfidfModel()

    # 读入测试集
    logger.info("read dev_sentences from: %s" % input_file_path)
    dev_sentences = read_file(input_file_path)

    logger.info("test is running, result will write to %s" % output_file_path)
    with open(output_file_path, 'w', encoding='utf-8') as file_result:
        for i in range(0, len(dev_sentences)):
            top_answers = ss.similarity(dev_sentences[i], top=conf.top)
            logger.info(top_answers)
            top_answers_index = [i[0] for i in top_answers]
            answer_candidates = []
            for a in top_answers_index:
                mid_answer = answers[a]
                answer_candidates.append((mid_answer, len(mid_answer)))
            logger.info(answer_candidates)
            answer = sorted(answer_candidates, key=lambda x: x[1],
                            reverse=True)[0][0]
            file_result.write(str(answer.strip(" ,，")) + '\n')
    logger.info("run prediction success.")
