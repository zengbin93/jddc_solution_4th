# coding=utf-8

import jieba
from zb.tools.file_tools import read_file
from zb.tools.file_tools import write_file


def tokenize_with_jieba(input_file, stopwords_file=None,
                        for_search=False, output_file=None):
    sentences = read_file(input_file)
    sentences_segs = []
    for sentence in sentences:
        if for_search:
            seg_list = jieba.cut_for_search(sentence)
        else:
            seg_list = jieba.cut(sentence)
        sentences_segs.append(list(seg_list))

    # 如果传入了stopwords_file文件，去停用词
    if stopwords_file:
        stopwords = read_file(stopwords_file)
        segs = []
        for seg_list in sentences_segs:
            results = []
            for seg in seg_list:
                if seg in stopwords:
                    continue
                results.append(seg)
            segs.append(results)
    else:
        segs = sentences_segs

    if output_file:
        segs_out = [str(x) for x in segs]
        write_file(output_file, segs_out,
                   mode='w', encoding='utf-8')
    return segs
