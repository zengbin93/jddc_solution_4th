# coding: utf-8

import jieba
import os
import re
import sys
from pprint import pprint
import pexpect
import traceback
import time


file_path = os.path.split(os.path.realpath(__file__))[0]
print("temp file will saved in %s" % file_path)


def read_questions(file):
    """读取questions测试集 - 完全多轮模式，即 QQ + A"""
    q_sep = "<q>"

    with open(file, 'r') as f:
        test_q = [x.strip("\n") for x in f.readlines()]
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
        if multi_turn[i - 1] == "":
            test_set.append(multi_turn[i])
        else:
            q = multi_turn[i - 1] + q_sep + multi_turn[i]
            test_set.append(q)
    return test_set


def jieba_tokenize(text, for_search=False):
    if for_search:
        tokens = list(jieba.cut_for_search(text))
    else:
        tokens = list(jieba.cut(text))
    return tokens


def run_prediction_old(input_file, output_file, clear=False, rq=True):
    print("read raw questions from %s" % input_file)
    q_set = read_questions(input_file)
    if rq:
        q_set = [" ".join(jieba_tokenize(q)[::-1])+'\n' for q in q_set]
    else:
        q_set = [" ".join(jieba_tokenize(q)) + '\n' for q in q_set]
    temp_file = os.path.join(file_path, "test_q.txt")
    middle_file = os.path.join(file_path, "answers.txt")

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.writelines(q_set)
    pprint(q_set)

    sh_script = "/bin/bash " + os.path.join(file_path, "t2t_predict.sh")
    child = pexpect.spawn(sh_script, encoding='utf-8',
                          timeout=72000, logfile=sys.stdout)
    print("execute predict: %s" % sh_script)
    child.read()

    with open(middle_file, 'r') as f:
        answers = [x.replace(" ", "") for x in f.readlines()]
    print("save answers to %s" % output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(answers)

    # 清理中间结果

    if clear:
        os.remove(temp_file)
        os.remove(middle_file)


# 通过命令行交互模式进行预测
# --------------------------------------------------------------------
def parse_answers(res):
    results = res.split("\r\n")
    answers = []
    for r in results:
        if "Score:" in r:
            r = r.replace("INFO:tensorflow:", "")
            a, s = r.split('\t')
            answer = a.strip("\"").replace(" ", "")
            score = s.replace("Score:", "").strip()
            answers.append((answer, float(score)))
    return answers


def run_prediction(input_file, output_file, rq=True):
    sh_script = "/bin/bash " + os.path.join(file_path, "t2t_predict.sh")
    child = pexpect.spawn(sh_script, encoding='utf-8',
                          timeout=72000, logfile=sys.stdout)
    try:
        print("execute predict interactive: %s" % sh_script)
        print("init ...")
        child.sendline("init ...")
        time.sleep(100)
        child.read_nonblocking(size=9999999, timeout=10)
        child.read_nonblocking(size=9999999, timeout=10)

        print("read raw questions from %s" % input_file)
        q_set = read_questions(input_file)

        answers = []

        for i, q in enumerate(q_set):
            print(i, "=" * 88)
            print("q:", q)
            q_tokens = jieba_tokenize(q)
            if rq:
                q_tokens = q_tokens[::-1]
            q_input = " ".join(q_tokens)
            child.sendline(q_input)
            time.sleep(10)
            y = child.read_nonblocking(size=9999999, timeout=1)
            a = parse_answers(y)
            # a = [a[0][0]] + [x[0] for x in a[1:] if float(x[1]) > -3]
            # a = sorted(a, key=lambda x: len(x), reverse=True)
            answer = re.sub("(#E-s\[数字x\])", "", a[0][0])
            if answer == "":
                answer = "请问还有其他还可以帮到您的吗?"
            print("a:", answer)
            answers.append(answer)

        print("save answers to %s" % output_file)
        answers = [x.strip(" \n") + "\n" for x in answers]
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(answers)
    except:
        traceback.print_exc()
        child.close()

    # 关闭交互进程
    child.close()





