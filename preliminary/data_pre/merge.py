# coding: utf-8
import codecs
import logging

from data_pre.config import BaseConf

conf = BaseConf()

files = {
    "chat": conf.file_chat,
    "questions": conf.file_qaqaq,
    "answers": conf.file_a,
}


def data_processing(files):
    """

    :param files: 三个文件
            {
            "chat": "chat.txt",
            "questions": "questions.txt",
            "answers": "answers.txt",
            }
    :return:
    """
    with codecs.open(files['questions'], mode="w", encoding="utf-8") as wfquestion:
        with codecs.open(files['answers'], mode="w", encoding="utf-8") as wfanswer:
            try:
                wfquestion.truncate()
                wfanswer.truncate()
            except Exception as e:
                logging.info("data_processing:clear txt error:", e)
                logging.exception(e)
            finally:
                wfquestion.close()
                wfanswer.close()

    question = ''
    answer = ''
    QAQAQ = ''
    countQuestion = 0
    countAnswer = 0
    # sessionId = "00029c51f92e8f34250d6af329c9a8df"  # 第一行的sessionID
    with codecs.open(files['chat'], mode='r', encoding="utf-8") as rf:
        try:
            line = rf.readline()
            sessionId = line.strip('\r\n').split("\t")[0]  # 第一行的sessionID
            while line:
                splitline = line.strip('\r\n').split("\t")
                if sessionId == splitline[0]:
                    with codecs.open(files['questions'],
                                     mode="a", encoding="utf-8") as wf_question:
                        with codecs.open(files['answers'],
                                         mode="a", encoding="utf-8") as wf_answer:
                            try:
                                if splitline[2] == '0':
                                    if countQuestion == 3 and countAnswer == 2:
                                        wf_question.write(QAQAQ + "\n")
                                        wf_answer.write(answer + "\n")
                                        question = ''
                                        answer = ''
                                        QAQAQ = ''
                                        countQuestion = 0
                                        countAnswer = 0

                                    if answer != '':
                                        # answer = answer.strip(',')
                                        # wf_question.write(answer)
                                        QAQAQ = QAQAQ + answer
                                        answer = ''
                                        countAnswer = countAnswer + 1
                                    question = question + splitline[6] + ','

                                elif splitline[2] == '1':
                                    if question != '':
                                        # question = question.strip(',')
                                        # wf_question.write(question)
                                        QAQAQ = QAQAQ + question
                                        question = ''
                                        countQuestion = countQuestion + 1
                                    answer = answer + splitline[6] + ','

                            except Exception as e:
                                logging.error("data_processing:write into txt failure", e)
                                logging.exception(e)
                            finally:
                                wf_question.close()
                                wf_answer.close()

                else:
                    sessionId = splitline[0]
                    question = ''
                    answer = ''
                    QAQAQ = ''
                    countQuestion = 0
                    countAnswer = 0
                    continue

                line = rf.readline()

        except Exception as e:
            logging.exception(e)
        finally:
            rf.close()


if __name__ == "__main__":
    data_processing(files)
