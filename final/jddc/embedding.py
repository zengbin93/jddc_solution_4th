# coding: utf-8
from jddc.config import EmbeddingConfig
from jddc.utils import read_from_pkl

conf = EmbeddingConfig()


def load_s2v_model():
    return read_from_pkl(conf.pkl_s2v)



