# coding: utf-8

from tensor2tensor import problems
import tensorflow as tf
import numpy as np
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
# 导入problem定义
from . import jddc

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = '/home/team55/notespace/data/t2t/data'
# Create hparams and the model
model_name = "transformer"
hparams_set = "transformer_base"
checkpoint_dir = '/home/team55/notespace/data/t2t/model_gf'
ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)

problem_name = 'jddc'
jddc_problem = problems.problem(problem_name)
# Get the encoders from the problem
encoders = jddc_problem.feature_encoders(data_dir)


hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir,
                                     problem_name=problem_name)
jddc_model = registry.model(model_name)(hparams, Modes.EVAL)


# Setup helper functions for encoding and decoding
def encode(input_str):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))


def predict_one(inputs):
    encoded_inputs = encode(inputs)
    print('\nencoded_inputs:')
    print(encoded_inputs)

    with tfe.restore_variables_on_create(ckpt_path):
        model_output = jddc_model.infer(encoded_inputs)["outputs"]
        print('\nmodel_output:')
        print(model_output)
    return decode(model_output)



