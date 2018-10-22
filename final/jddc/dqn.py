# coding: utf-8
"""
解决方案：DQN + IR

借助DQN来提高IR的性能，对IR的候选结果排序
====================================================================
"""
from collections import deque
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from jddc.utils import cal_bleu_score, n_grams


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)


def choose_action(model, state, epsilon, use_cuda=False):
    """根据当前state，选择action"""
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            state = Variable(state)
        if use_cuda:
            state = state.cuda()
        q_value = model(state)
        action = q_value.max(1)[1].data[0]
    else:
        action = random.randrange(model.num_actions)
    return action


class Environment(object):
    """强化学习模型 - 【环境】"""

    def __init__(self, sessions, s2v_model):
        # 每一个session都是Session对象
        self.sessions = sessions
        # sentence2vec
        self.s2v_model = s2v_model

        self.num_state_features = 300
        self.num_actions = None
        self.meta = {"name": "environment", "version": "0.1.0"}

        # bleu奖励的范围
        self.reward_range = [0, 1]

        self.done = False
        self.state = None
        self.multi_qa = None
        self.turn_i = None
        self.total_turn = None

    @staticmethod
    def _get_session_qa(session):
        # 获取单个session的中所有轮次的QA
        qas = session.multi_qa(merge_q=True)
        multi_qa = []
        for x in qas:
            q = "<q>".join(x[2].split("<q>")[-2:])
            a = x[3]
            multi_qa.append((q, a))
        return multi_qa

    def reset(self):
        """重置环境"""
        self.done = False
        session = random.sample(self.sessions, 1)[0]
        self.multi_qa = self._get_session_qa(session)
        self.total_turn = len(self.multi_qa)
        self.turn_i = 0
        q_tokens = n_grams(self.multi_qa[self.turn_i][0], n=3)
        self.state = self.s2v_model.infer_vector(q_tokens)
        return self.state, q_tokens

    def step(self, action):
        """对Agent的action做出响应，主要包括：
        1）计算奖励
        2）更新state
        """
        # 计算动作奖励
        ref_a = self.multi_qa[self.turn_i][1]
        award = cal_bleu_score(action, ref_a)

        # 更新state
        self.turn_i += 1
        if self.turn_i >= len(self.multi_qa) - 1:
            q_tokens = None
            self.state = np.zeros(self.s2v_model.vector_size)
            self.done = True
        else:
            q_tokens = n_grams(self.multi_qa[self.turn_i][0], n=3)
            self.state = self.s2v_model.infer_vector(q_tokens)
            self.done = False
        return self.state, award, self.done, q_tokens


# 训练DQN
# --------------------------------------------------------------------

class DQNTrainer(object):
    """DQN网络训练"""
    def __init__(self, sessions, s2v_model, ir_model, use_cuda=False):
        self.env = Environment(sessions=sessions, s2v_model=s2v_model)
        self.dqn_model = DQN(num_inputs=300, num_actions=15)
        self.replay_buffer = ReplayBuffer(1000)
        self.optimizer = optim.Adam(self.dqn_model.parameters())
        self.ir_model = ir_model

        self.use_cuda = use_cuda

        # 参数
        self.num_frames = 1000000
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.9

        # 运行中间结果收集
        self.losses = []
        self.all_rewards = [0]

        if use_cuda:
            self.dqn_model = self.dqn_model.cuda()

    def compute_loss(self, batch_size):
        state, action, reward, next_state, done = \
            self.replay_buffer.sample(batch_size)
        gamma = self.gamma

        with torch.no_grad():
            state = Variable(torch.FloatTensor(np.float32(state)))
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))
            action = Variable(torch.LongTensor(action))
            reward = Variable(torch.FloatTensor(reward))
            done = Variable(torch.FloatTensor(done))

        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()
            self.dqn_model = self.dqn_model.cuda()

        q_values = self.dqn_model(state)
        next_q_values = self.dqn_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def run(self):
        env = self.env
        num_frames = self.num_frames
        replay_buffer = self.replay_buffer
        batch_size = self.batch_size

        # Epsilon greedy exploration
        epsilon_decay = 0.9
        epsilon_min = 0.1

        episode_reward = 0
        state, q_tokens = env.reset()
        for frame_idx in tqdm(range(1, num_frames + 1), ncols=100, desc="train dqn"):
            # 获取候选答案
            q = q_tokens[0][:2] + "".join([x[-1] for x in q_tokens])
            candidates = self.ir_model.get_candidates(q, top=15)

            # dqn模型选择action
            action = choose_action(self.dqn_model, state, self.epsilon,
                                   use_cuda=self.use_cuda)
            # 环境根据action做出响应
            next_state, reward, done, q_tokens = env.step(candidates[action])
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                self.all_rewards.append(episode_reward)
                mean_reward = round(episode_reward / env.total_turn, 4)
                print("%s | episode_total_reward:%.4f; mean_reward:%.4f" % (
                    str(datetime.now()), episode_reward, mean_reward
                ))
                state, q_tokens = env.reset()
                # 每10000轮降低epsilon一次
                if frame_idx % 10000 == 0:
                    if self.epsilon > epsilon_min:
                        self.epsilon = self.epsilon * epsilon_decay
                    print("current epsilon:", self.epsilon)
                episode_reward = 0

            if len(replay_buffer) > batch_size:
                loss = self.compute_loss(batch_size)
                self.losses.append(loss.data[0])
            if frame_idx % 1000 == 0:
                loss = self.compute_loss(batch_size)
                print("%i loss is " % frame_idx, loss.data[0])


