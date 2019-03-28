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
import numpy as np
from pickle import dump,load



# - 步骤2：训练数据读取和转换

init_weight_wv= load(open('init_weight.pickle','rb'))
word2idx = load(open('word2idx.pickle', 'rb'))
idx2word = load(open('idx2word.pickle', 'rb'))

# 读取数据，将格式进行转换为带四种标签 S B M E
input_file = './data/icwb2-data/training/msr_training.utf8'
output_file = './data/icwb2-data/training/msr_training.tagging.utf8'


# 用于字符标记的4个标签：B（开始），E（结束），M（中），S（单）

def character_tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/S ")
            else:
                output_data.write(word[0] + "/B ")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word)-1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()

character_tagging(input_file, output_file)



# 定义'U'为未登陆新字, 空格为两头padding用途，并增加两个相应的向量表示
char_num = len(init_weight_wv)
idx2word[char_num] = u'U'
word2idx[u'U'] = char_num
idx2word[char_num+1] = u' '
word2idx[u' '] = char_num+1

init_weight_wv.append(np.random.randn(100,))
init_weight_wv.append(np.zeros(100,))


# In[21]:


# 分离word 和 label
with open(output_file) as f:
    lines = f.readlines()
    train_line = [[w[0] for w in line.split()] for line in lines]
    train_label = [w[2] for line in lines for w in line.split()]


# In[17]:


# 文档转数字list
def sent2num(sentence, word2idx = word2idx, context = 7):
    predict_word_num = []
    for w in sentence:
        # 文本中的字如果在词典中则转为数字，如果不在则设置为'U
        if w in word2idx:
            predict_word_num.append(word2idx[w])
        else:
            predict_word_num.append(word2idx[u'U'])
    # 首尾padding
    num = len(predict_word_num)
    pad = int((context-1)*0.5)
    for i in range(pad):
        predict_word_num.insert(0,word2idx[u' '] )
        predict_word_num.append(word2idx[u' '] )
    train_x = []
    for i in range(num):
        train_x.append(predict_word_num[i:i+context])
    return train_x


# In[53]:


# 输入字符list，输出数字list
sent2num(train_line[0])


# In[60]:


# 将所有训练文本转成数字list
train_word_num = []
for line in train_line:
    train_word_num.extend(sent2num(line))


# In[62]:


print(len(train_word_num))
print(len(train_label))

# In[64]:


dump(train_word_num, open('train_word_num.pickle', 'wb'))
#train_word_num = load(open('train_word_num.pickle','rb'))

dump(train_label, open('train_label.pickle', 'wb'))
dump(sent2num, open('sent2num.pickle', 'wb'))

# In[22]:



def main():
    pass


if __name__ == '__main__':
    main()