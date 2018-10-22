#! /usr/local/python3.6.5/bin//python3.6
# -*- coding: utf-8 -*-


from jddc.config import DQNConfig
from jddc.embedding import load_s2v_model
from jddc.tfidf import load_tfidf_ir_model
import jddc.utils as u
from jddc.dqn import *


conf = DQNConfig()
print(str(conf))
refresh = True

if refresh:
    print("refresh model")
    sessions = u.read_from_pkl(conf.pkl_sessions)
    sessions = random.sample(sessions, 100000)
    s2v_model = load_s2v_model()
    ir_model = load_tfidf_ir_model()
    trainer = DQNTrainer(sessions, s2v_model, ir_model, use_cuda=True)
    trainer.run()
    u.save_to_pkl(file=conf.pkl_dqn_ir, data=trainer)
else:
    trainer = u.read_from_pkl(conf.pkl_dqn_ir)
    # 这里可以修改一些参数
    trainer.run()
