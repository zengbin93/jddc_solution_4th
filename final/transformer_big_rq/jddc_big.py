# coding: utf-8

import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


@registry.register_problem
class JddcBig(text_problems.Text2TextProblem):
    """JDDC多轮对话"""

    @property
    def approx_vocab_size(self):
        return 2 ** 18  # ~262k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data
        # into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number
        of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 3999,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir=None, dataset_split=None):
        del tmp_dir
        del dataset_split

        file_qa = os.path.join(data_dir, "qa_pairs_rq.tsv")

        with open(file_qa, 'r') as f:
            qa_pairs = f.readlines()

        for row in qa_pairs:
            q, a = row.strip("\n").split("\t")
            # yield {
            #     "inputs": q.replace(" ", ""),
            #     "targets": a.replace(" ", ""),
            # }
            yield {
                "inputs": q,
                "targets": a,
            }
