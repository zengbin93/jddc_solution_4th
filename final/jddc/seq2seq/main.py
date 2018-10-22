# coding: utf-8
"""

====================================================================
"""

from .checkpoint import Checkpoint
from .predictor import Predictor

from jddc.config import Seq2SeqConfig
from jddc.datasets import read_test_questions02
import jddc.utils as u

conf = Seq2SeqConfig()

# 指定提交所用的模型
latest_checkpoint_path = ""


def run_prediction(input_file, output_file, rq=False):
    print("load model from %s" % latest_checkpoint_path)
    resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
    model = resume_checkpoint.model.cpu()
    model.decoder.use_cuda = False
    src_vocab = resume_checkpoint.input_vocab
    tgt_vocab = resume_checkpoint.output_vocab

    predictor = Predictor(model, src_vocab, tgt_vocab, use_cuda=False)
    test_q = read_test_questions02(input_file)

    answers = []
    for i, q in enumerate(test_q, 1):
        print(i, "=" * 68 + "\n", "question:", q)
        q_tokens = u.jieba_tokenize(q)
        if rq:
            q_tokens = q_tokens[::-1]
        results = predictor.predict_n(q_tokens, 6)
        candidates = []
        print("answer:\n")

        for idx, x in enumerate(results[1:], 2):
            candidate = "".join(x)
            candidate = candidate.replace("<eos>", "").replace("<pad>", "")
            candidates.append(candidate)
            print(idx, candidate)
        # 选最长
        candidates = sorted(candidates, key=lambda a: len(a), reverse=True)
        answers.append(candidates[0])
    u.write_file(file=output_file, content=answers, mode='w', encoding='utf-8')

