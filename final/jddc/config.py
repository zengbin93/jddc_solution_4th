import os
from jddc.utils import insure_folder_exists
from jddc.utils import AttrDict


class BaseConfig(object):
    """Config基类：基础配置参数"""
    # 中间数据和结果存储目录
    base_path = "/home/team55/notespace/data"
    # base_path = "/submitwork/data"
    # base_path = "C:\ZB\git_repo\JDDC\data"
    insure_folder_exists(base_path)
    temp_path = os.path.join(base_path, "temp")
    insure_folder_exists(temp_path)

    # 用于分词的自定义词典
    file_dict = os.path.join(base_path, "jddc_dict.txt")

    # 原始数据
    raw_path = "/mnt/dataset/finalTrainData"
    file_chat = os.path.join(raw_path, 'chat.txt')
    file_ware = os.path.join(raw_path, 'ware.txt')
    file_order = os.path.join(raw_path, 'order.txt')
    file_user = os.path.join(raw_path, 'user.txt')

    # 测试数据
    test_data_path = "/mnt/dataset/finalTestData"
    file_test_q = os.path.join(test_data_path, '单轮questions50+多轮sessions5.txt')
    file_test_a = os.path.join(test_data_path, '标准答案集单轮50+多轮5.txt')

    # 其他：日志等
    file_stopwords = os.path.join(base_path, "stopwords_v0.txt")
    log_file = os.path.join(base_path, 'jddc_final.log')
    cmd_log = True


class PreConfig(BaseConfig):
    """数据预处理相关配置"""

    def __init__(self):
        super(PreConfig, self).__init__()
        self.prepared_path = os.path.join(self.base_path, "prepared")
        insure_folder_exists(self.prepared_path)

        # temp files
        self.file_chat_pred = os.path.join(self.temp_path, "chat_pred.txt")
        self.file_chat_splited = os.path.join(self.temp_path, 'chat_splited.txt')
        self.file_session_parsed = os.path.join(self.temp_path, 'sessions.txt')
        self.pkl_sessions = os.path.join(self.temp_path, "all_sessions.pkl")

        # 用于训练词向量的数据
        self.file_texts_for_embedding = os.path.join(self.prepared_path, "texts_for_embedding.txt")

        # 单轮对话训练集
        self.file_qa_10000 = os.path.join(self.prepared_path, "qa_10000.txt")
        self.file_qa_100000 = os.path.join(self.prepared_path, "qa_100000.txt")

        self.file_q_tokens = os.path.join(self.prepared_path, "q_tokens.txt")

        # 多轮对话训练集
        self.pkl_mqa_100000 = os.path.join(self.prepared_path, "multi_qa_100000.pkl")
        self.pkl_mqa_10000 = os.path.join(self.prepared_path, "multi_qa_10000.pkl")
        self.pkl_mqa_1000 = os.path.join(self.prepared_path, "multi_qa_1000.pkl")


class EmbeddingConfig(PreConfig):
    """文本表达相关配置"""

    def __init__(self):
        super(EmbeddingConfig, self).__init__()
        self.embedding_path = os.path.join(self.base_path, "embedding")
        insure_folder_exists(self.embedding_path)

        # Word2vec
        self.pkl_w2v = os.path.join(self.embedding_path, "word2vec.model.pkl")

        # Sentence2vec
        self.pkl_s2v = os.path.join(self.embedding_path, "sentence2vec.model.pkl")


class TestConfig(EmbeddingConfig):
    """用于测试的配置参数"""

    def __init__(self):
        super(TestConfig, self).__init__()
        self.test_path = os.path.join(self.base_path, "test")
        insure_folder_exists(self.test_path)
        self.file_qa1000 = os.path.join(self.test_path, "qaqaq_a_sample_1000.txt")
        self.file_qa10000 = os.path.join(self.test_path, "qaqaq_a_sample_10000.txt")
        self.file_qa50000 = os.path.join(self.test_path, "qaqaq_a_sample_50000.txt")


class Seq2SeqConfig(TestConfig):
    """Seq2Seq模型参数配置"""
    use_cuda = True
    teacher_forcing_ratio = 0.5

    # encoder & decoder
    hidden_size = 256
    n_layers = 5
    bidirectional = True
    max_len = 300
    rnn_cell = 'lstm'

    encoder_params = AttrDict()
    encoder_params['hidden_size'] = hidden_size
    encoder_params['n_layers'] = n_layers
    encoder_params['bidirectional'] = bidirectional
    encoder_params['max_len'] = max_len
    encoder_params['rnn_cell'] = rnn_cell
    encoder_params['variable_lengths'] = True
    encoder_params['input_dropout_p'] = 0
    encoder_params['dropout_p'] = 0.3

    decoder_params = AttrDict()
    decoder_params['hidden_size'] = hidden_size*2 if bidirectional else hidden_size
    decoder_params['n_layers'] = n_layers
    decoder_params['bidirectional'] = bidirectional
    decoder_params['max_len'] = max_len
    decoder_params['rnn_cell'] = rnn_cell
    decoder_params['use_attention'] = True
    decoder_params['use_cuda'] = use_cuda
    decoder_params['input_dropout_p'] = 0
    decoder_params['dropout_p'] = 0.3

    def __init__(self):
        super(Seq2SeqConfig, self).__init__()

        # 模型存储目录
        self.s2s_path = os.path.join(self.base_path, "seq2seq")
        insure_folder_exists(self.s2s_path)

        self.checkpoints_path = os.path.join(self.s2s_path, "checkpoints")
        insure_folder_exists(self.s2s_path)

        self.file_qa_pairs = os.path.join(self.s2s_path, "qa_pairs.pkl")
        self.file_train = os.path.join(self.s2s_path, "train.tsv")
        # 翻转QQ分词结果
        self.file_train_rq = os.path.join(self.s2s_path, "train_reverse_q.tsv")


class TfidfConfig(TestConfig):
    """Tfidf检索模块配置"""

    def __init__(self):
        super(TfidfConfig, self).__init__()
        self.tfidf_path = os.path.join(self.base_path, 'tfidf')
        insure_folder_exists(self.tfidf_path)

        # 数据集
        self.pkl_tfidf_ir = os.path.join(self.tfidf_path, "tfidf_ir.model.middle.pkl")

        self.n = 3
        self.top = 5

    def __repr__(self):
        return "<TfidfConfig | model:%s; n:%s; top:%s>" % (
            self.pkl_tfidf_ir, str(self.n), str(self.top)
        )


class BM25Config(TestConfig):
    """BM25检索相关参数"""

    def __init__(self):
        super(BM25Config, self).__init__()
        self.bm25_path = os.path.join(self.base_path, "BM25")
        insure_folder_exists(self.bm25_path)

        # 数据集生成过程使用的随机数
        self.random_num = 1234

        self.top = 5  # 召回结果数量
        self.n = 3  # n-grams参数

        # 模型
        self.pkl_bm25 = os.path.join(self.bm25_path, "bm25_%i.model.small.pkl" % self.n)

    def __repr__(self):
        return "<BM25Config | model:%s; n:%s; top:%s; random_num:%s>" % (
            self.pkl_bm25, str(self.n), str(self.top), self.random_num
        )


class DQNConfig(TestConfig):
    """基于强化学习DQN的方法
    s2v模块、以BM25作为基础检索模块
    """

    def __init__(self):
        super(DQNConfig, self).__init__()
        self.dqn_path = os.path.join(self.base_path, "DQN")
        insure_folder_exists(self.dqn_path)

        self.use_cuda = True
        self.num_inputs = 300
        self.num_actions = 15

        # 模型文件
        self.pkl_dqn_ir = os.path.join(self.dqn_path, "dqn.model.pkl")

    def __repr__(self):
        return "<DQNConfig | model:%s; use_cuda:%s; num_inputs:%s; num_actions:%s>" % (
            self.pkl_dqn_ir, str(self.use_cuda), str(self.num_inputs), self.num_actions
        )



