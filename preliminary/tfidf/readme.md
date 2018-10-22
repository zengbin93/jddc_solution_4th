## TFIDF基线模型 + top选最长

> 当前最佳得分模型

tfidf模型，主要思路是：首先计算用户的问题与问题库中的问题的相似度并选出top15的相似问题，然后去问题库对应的答案库中找出这15个问题对应的答案，
以此作为回答用户问题的候选答案。代码参考：https://github.com/WenDesi/sentenceSimilarity
运行于python3.6环境下。

### 用TFIDF方法做检索，其基本过程如下：
假设输入查询为query，query中包含n个词语，分别是q1、q2、…、qn；语料库为D，包含若干个句子。

step 1. 对语料库D中的所有句子进行分词；

step 2. 构建bag-of-word模型，给每个词一个id；

step 3. 计算语料库D中所有词的tfidf值；

step 4. 计算语料库D中所有句子的tfidf向量表达；

	对于任一句子，其tfidf向量表达是句子中所有词的tfidf值构成的向量，保留词的先后顺序。

step 5. 对query分词，生成tfidf向量表达，计算该向量与语料库D中所有句子向量的相似度，选取top N作为检索结果。


