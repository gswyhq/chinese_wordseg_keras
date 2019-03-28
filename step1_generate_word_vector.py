#!/usr/bin/python3
# coding=utf-8


# ### 基于深度学习的中文分词尝试
# - 基于word2vec + 神经网络进行中文分词
#     - 步骤1：使用的是sogou的语料库建立初始的字向量。
#     - 步骤2：读入有标注的训练语料库，处理成keras需要的数据格式。
#     - 步骤3：根据训练数据建模，使用CNN方法
#     - 步骤4：读入无标注的检验语料库，用CNN模型进行分词标注
#     - 步骤5：检查最终的效果
# - 参考资料:[中文分词资源](http://www.52nlp.cn/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%85%A5%E9%97%A8%E4%B9%8B%E8%B5%84%E6%BA%90)
#     [中文分词标注法](http://www.52nlp.cn/the-character-based-tagging-method-of-chinese-word-segmentation) [word2vec原理](http://suanfazu.com/t/word2vec-zhong-de-shu-xue-yuan-li-xiang-jie-duo-tu-wifixia-yue-du/178) [基于word2vec的中文分词](http://blog.csdn.net/itplus/article/details/17122431)

# - 步骤1：先用sogou语料库生成中文的单字向量，以备后用

import codecs
import sys
from os import path
import os
import re
import pickle
import numpy as np
import pandas as pd
import itertools
import nltk
from nltk.probability import FreqDist
from pickle import dump,load
from gensim.models import word2vec

# SogouC.reduced.zip; https://pan.baidu.com/s/1pLrORJt
rootdir = './data/SogouC.reduced/Reduced'
dirs = os.listdir(rootdir)
dirs = [path.join(rootdir,f) for f in dirs if f.startswith('C')]

def load_txt(x):
    with open(x) as f:
        res = [t for t in f.readlines()]
        return ''.join(res)


# In[4]:

def to_df(dirs):
    text_t = {}
    for i, d in enumerate(dirs):
        files = os.listdir(d)
        files = [path.join(d, x) for x in files if x.endswith('txt') and not x.startswith('.')]
        text_t[i] = [load_txt(f) for f in files]

    # to dataframe

    flen = [len(t) for t in text_t.values()]


    # In[7]:


    labels = np.repeat(list(text_t.keys()),flen)


    # In[8]:


    # flatter nested list

    merged = list(itertools.chain.from_iterable(text_t.values()))


    # In[9]:


    df = pd.DataFrame({'label': labels, 'txt': merged})
    df.head()
    df['seg_word'] = df.txt.map(cutchar)

    return df

# In[10]:


# cut character
def cutchar(x):
    words = list(x)
    return ' '.join(words)

df = to_df(dirs)
dump(df, open('./df.pickle', 'wb'))

# df = load(open('df.pickle','rb'))

# 探索 转成nltk需要格式,建立list
txt = df['seg_word'].values
txtnltk = []
for sent in txt:
    temp = [w for w in sent.split()]
    txtnltk.extend(temp)

# nltk
corpus = nltk.Text(txtnltk)

# 词频
fdist = FreqDist(corpus)
# w = fdist.keys()
# v = fdist.values()
w, v = [], []
for k1, v1 in fdist.items():
    w.append(k1)
    v.append(v1)
freqdf = pd.DataFrame({'word':w,'freq':v})
freqdf.sort_values('freq',ascending =False, inplace=True)
freqdf['idx'] = np.arange(len(v))
freqdf.head()


# In[14]:
word2idx = {c:i for c, i in zip(freqdf.word, freqdf.idx)}
idx2word = dict((i, c) for c, i in zip(freqdf.word, freqdf.idx))

print('保存word2idx到文件')
dump(word2idx, open('word2idx.pickle', 'wb'))
print('保存idx2word到文件')
dump(idx2word, open('idx2word.pickle', 'wb'))

# In[23]:


# 保持字符串，为生成词向量准备
all_news_wv = []
for news in txt:
    all_news_wv.append([x for x in news.split() ])


# In[24]:


# word2vec
def trainW2V(corpus, epochs=50, num_features = 100,
             min_word_count = 1, num_workers = 4,
             context = 10, sample = 1e-5):
    global w2v
    w2v = word2vec.Word2Vec(workers = num_workers,
                          sample = sample,
                          size = num_features,
                          min_count=min_word_count,
                          window = context)
    np.random.shuffle(corpus)
    w2v.build_vocab(corpus)
    for epoch in range(epochs):
        print(epoch, ' ')
        np.random.shuffle(corpus)
        w2v.train(corpus, epochs=w2v.iter,total_examples=w2v.corpus_count)
        w2v.alpha *= 0.9
        w2v.min_alpha = w2v.alpha
    print("Done.")


# In[25]:


# word2vec
trainW2V(all_news_wv)


# In[26]:


# 保存词向量lookup矩阵，按idx位置存放。目的是保存词频，也可以直接使用w2v.index2word
init_weight_wv = []
for i in range(freqdf.shape[0]):
    init_weight_wv.append(w2v[idx2word[i]])


# In[15]:

print('保存字向量到文件')
dump(init_weight_wv, open('init_weight.pickle', 'wb'))


def main():
    pass


if __name__ == '__main__':
    main()