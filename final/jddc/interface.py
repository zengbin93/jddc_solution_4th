# coding: utf-8
"""
接口模块
====================================================================
"""


class BaseIR(object):
    """IR基类"""

    def __init__(self, name=None):
        self.name = name

    def get_candidates(self, q_tokens, top=10):
        """获取候选答案

        :param q_tokens: list of str
            分好词的question
        :param top: int
        :return: list
            top个候选答案
        """
        raise NotImplementedError


