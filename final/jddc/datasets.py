# coding: utf-8
"""
数据集构建
====================================================================
"""

from tqdm import tqdm
import random
import re

from jddc.utils import read_file

q_sep = "<q>"
max_len = 200

random.seed(6868)


def create_dataset01(sessions):
    """创建数据集

    方案描述：

    1. 从全部对话数据集中抽样1000000个session
    2. 每个session至少随机抽取一个QA对
    3. 提取每个session中的QQQ + A模式数据

    """
    # all_sessions = read_from_pkl(conf.pkl_sessions)
    # sessions = random.sample(all_sessions, 1000000)
    data_q = []
    data_a = []
    for sess in tqdm(sessions, desc="create_dataset01"):
        # Q + A
        qas = sess.qas_merged
        if qas[0][0] == "1":
            qas = qas[1:]
        if qas[-1][0] == "0":
            qas = qas[:-1]
        assert len(qas) % 2 == 0
        for i in range(0, len(qas), 2):
            # 随机判断是否要新增QA
            if random.choice([True, False]):
                data_q.append(qas[i][1].replace("\t", " "))
                data_a.append(qas[i+1][1].replace("\t", " "))
            # 是否直接退出
            if random.choice([True, False]):
                break

        # QQQ + A
        q, a = sess.qaqaq_a
        if 2 < len(a) < 100:
            q = re.sub("(\t)| ", "", q)
            q_split = q.split("<s>")
            if len(q_split) == 5:
                q_merged = q_sep.join([q_split[0], q_split[2], q_split[4]])
                data_q.append(q_merged.replace("\t", " "))
                data_a.append(a.replace("\t", " "))
    print("created qa pairs num is ", len(data_q))
    return data_q, data_a


def create_dataset02(sessions):
    """创建数据集 - 方案二

    方案描述：

    1. 从全部对话数据集中抽样1000000个session
    2. 每个session随机采样若干个QA对
    3. 提取每个session中的QQQ + A模式数据
    """
    # all_sessions = read_from_pkl(conf.pkl_sessions)
    # sessions = random.sample(all_sessions, 1000000)
    data_q = []
    data_a = []
    for sess in tqdm(sessions, desc="create_dataset02"):
        # 获取单个session的中所有轮次的QA
        qas = sess.qas_merged
        if qas[0][0] == "1":
            qas = qas[1:]
        if qas[-1][0] == "0":
            qas = qas[:-1]
        assert len(qas) % 2 == 0
        multi_qa = []
        multi_q = []
        for i in range(0, len(qas), 2):
            multi_q.append(qas[i][1].replace("\t", ""))
            q = q_sep.join(multi_q[-2:])
            a = qas[i+1][1].replace("\t", " ")
            # 仅保留answer长度小于300的QA对
            if len(a) < 300:
                multi_qa.append((q, a))

        # 随机采样若干个QA对
        num = random.choice(range(len(multi_qa)))
        qa_selected = random.sample(multi_qa, num)
        for q, a in qa_selected:
            data_q.append(q)
            data_a.append(a)

        # QQQ + A
        q, a = sess.qaqaq_a
        if 2 < len(a) < 100:
            q = re.sub("(\t)| ", "", q)
            q_split = q.split("<s>")
            if len(q_split) == 5:
                q_merged = q_sep.join([q_split[0], q_split[2], q_split[4]])
                data_q.append(q_merged)
                data_a.append(a.replace("\t", " "))
    print("created qa pairs num is ", len(data_q))
    return data_q, data_a


def create_dataset03(sessions):
    """数据集构建方案三：
    step 1. 构建单轮数据集和多轮数据的session独立取样
    step 2. 单轮数据集 QQQ + A
    step 3. 多轮数据集 QQ + A
    """
    left = random.sample(sessions[:400000], 20000)
    right = random.sample(sessions[400000:], 20000)

    data_q = []
    data_a = []

    # 单轮数据集
    for sess in left:
        q, a = sess.qaqaq_a
        if 2 < len(a) < 100:
            q = re.sub("(\t)| ", "", q)
            q_split = q.split("<s>")
            if len(q_split) == 5:
                q_merged = q_sep.join([q_split[0], q_split[2], q_split[4]])
                data_q.append(q_merged.replace("\t", ""))
                data_a.append(a.replace("\t", " "))

    # 多轮数据集
    for sess in right:
        qas = sess.multi_qa(merge_q=True)
        for qa in qas:
            q, a = qa[2:4]
            # 随机判断是否要新增QA
            if random.choice([True, False]) and 3 < len(a) < 300:
                q = q_sep.join(q.split(q_sep)[-2:])
                data_q.append(q.replace("\t", ""))
                data_a.append(a.replace("\t", " "))
    questions, answers = data_q, data_a
    assert len(questions) == len(answers)
    print("created qa pairs num is ", len(questions))
    return questions, answers


def create_dataset04(sessions):
    """数据集构建方案四： 完全的多轮数据集
    step 1. 从sessions取样
    step 2. 多轮数据集 QQ + A
    """
    right = random.sample(sessions, 30000)

    data_q = []
    data_a = []

    # 多轮数据集
    for sess in right:
        qas = sess.multi_qa(merge_q=True)
        for qa in qas:
            q, a = qa[2:4]
            # 随机判断是否要新增QA
            if random.choice([True, False]) and 3 < len(a) < 300:
                q = q_sep.join(q.split(q_sep)[-2:])
                data_q.append(q.replace("\t", ""))
                data_a.append(a.replace("\t", " "))
    questions, answers = data_q, data_a
    assert len(questions) == len(answers)
    print("created qa pairs num is ", len(questions))
    return questions, answers


def create_dataset05(sessions, sample_n=30000, random_sample=False):
    """数据集构建方案五： 完全的多轮数据集 + 样本均衡（可选）
    step 1. 从sessions采样若干个session
    step 2. 提取每一个session中所有 QQ + A
    step 3. 均衡样本：第一个和最后一个QQ+A随机选取，中间的全部保留
        为什么要均衡样本？
        通过我的观察，发现绝大多好session的开头和结尾都是一样的，
        中间部分各有千秋；如果完全保留全部QQ+A，那么开头和结尾的
        样本占比就会更高，影响中间QQ的结果。
    """
    right = random.sample(sessions, sample_n)
    head_qa = []
    data_qa = []
    tail_qa = []

    # 多轮数据集
    for sess in tqdm(right, ncols=100, desc='create dataset05'):
        qas = sess.multi_qa(merge_q=True)

        # 获取session开头和结尾的对话
        head_q, head_a = qas[0][2:4]
        if 3 < len(head_a) < max_len and len(head_q) < max_len:
            head_qa.append((head_q, head_a))

        last_q = q_sep.join(qas[-1][2].split(q_sep)[-2:])
        last_a = qas[-1][3]
        if 3 < len(last_a) < max_len and len(last_q) < max_len:
            tail_qa.append((last_q, last_a))

        for qa in qas[1:-1]:
            q, a = qa[2:4]
            if 3 < len(a) < max_len and len(q) < max_len:
                q = q_sep.join(q.split(q_sep)[-2:])
                data_qa.append((q, a))
    if random_sample:
        # 随机采样
        head_qa = random.sample(head_qa, int(len(data_qa)/8))
        tail_qa = random.sample(tail_qa, int(len(data_qa)/8))
        print("sample head and tail qa num is ", len(head_qa+tail_qa))

    dataset = head_qa + tail_qa + data_qa
    questions = [x[0].replace("\t", "") for x in dataset]
    answers = [x[1].replace("\t", " ") for x in dataset]
    assert len(questions) == len(answers)
    print("created qa pairs num is ", len(questions))
    return questions, answers


def create_dataset06(sessions, sample_n=200000):
    """用QAQAQ + A构造 QQ+A数据集
    """
    sessions = random.sample(sessions, sample_n)
    data_q = []
    data_a = []
    for sess in sessions:
        q, a = sess.qaqaq_a
        q_splited = q.split("<s>")
        try:
            q = q_sep.join([q_splited[2], q_splited[4]])
            if 3 < len(a) < max_len and len(q) < max_len:
                data_q.append(q.replace("\t", ""))
                data_a.append(a.replace("\t", " "))
        except:
            continue
    return data_q, data_a


def create_dataset07(sessions, sample_n=30000):
    """数据集构建方案七： 完全的多轮数据集
    step 1. 从sessions采样若干个session
    step 2. 提取每一个session中所有 QQ + A
    step 3. 删除第一个和最后一个QQ+A
    """
    right = random.sample(sessions, sample_n)
    data_qa = []
    # 多轮数据集
    for sess in tqdm(right, ncols=100, desc='create dataset07'):
        qas = sess.multi_qa(merge_q=True)
        for qa in qas[1:-1]:
            q, a = qa[2:4]
            if 3 < len(a) < max_len and len(q) < max_len:
                q = q_sep.join(q.split(q_sep)[-2:])
                data_qa.append((q, a))
    questions = [x[0].replace("\t", "") for x in data_qa]
    answers = [x[1].replace("\t", " ") for x in data_qa]
    assert len(questions) == len(answers)
    print("created qa pairs num is ", len(questions))
    return questions, answers


# 读取测试数据
# --------------------------------------------------------------------
def read_test_questions01(file):
    """读取questions测试集"""
    test_q = read_file(file)
    single_turn = [x for x in test_q if "<s>" in x]
    multi_turn = [x for x in test_q if "<s>" not in x]

    test_set = []
    # 单轮测试数据
    for i in range(len(single_turn)):
        tq = single_turn[i]
        tq = re.sub("(\t)| ", "", tq)
        q_split = tq.split("<s>")
        if len(q_split) > 4:
            del q_split[3]
            del q_split[1]
        elif len(q_split) > 2:
            del q_split[1]
        else:
            pass
        tq = q_sep.join(q_split)
        test_set.append(tq)

    # 多轮测试数据
    for i in range(len(multi_turn)):
        # 遇到空行，跳过
        if multi_turn[i] == "":
            continue
        # 首行
        if i == 0:
            test_set.append(multi_turn[i])
            continue
        # 逐行处理（构造QQ）
        if multi_turn[i-1] == "":
            test_set.append(multi_turn[i])
        else:
            q = multi_turn[i-1] + q_sep + multi_turn[i]
            test_set.append(q)
    return test_set


def read_test_questions02(file):
    """读取questions测试集 - 完全多轮模式，即 QQ + A"""
    test_q = read_file(file)
    single_turn = [x for x in test_q if "<s>" in x]
    multi_turn = [x for x in test_q if "<s>" not in x]

    test_set = []
    # 单轮测试数据
    for i in range(len(single_turn)):
        tq = single_turn[i]
        q_split = tq.split("<s>")
        if len(q_split) == 5:
            q_split = [q_split[2], q_split[4]]
        elif len(q_split) == 3:
            q_split = [q_split[0], q_split[2]]
        else:
            q_split = q_split[-2:]
        tq = q_sep.join(q_split)
        test_set.append(tq)

    # 多轮测试数据
    for i in range(len(multi_turn)):
        # 遇到空行，跳过
        if multi_turn[i] == "":
            continue
        # 首行
        if i == 0:
            test_set.append(multi_turn[i])
            continue
        # 逐行处理（构造QQ）
        if multi_turn[i-1] == "":
            test_set.append(multi_turn[i])
        else:
            q = multi_turn[i-1] + q_sep + multi_turn[i]
            test_set.append(q)
    return test_set


def read_test_answers(file):
    return [x for x in read_file(file) if x != ""]



