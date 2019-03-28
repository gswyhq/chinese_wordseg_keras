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

import numpy as np
from pickle import dump,load


# - 步骤4：用test文本进行预测，评估效果

# In[25]:
sent2num = load(open('sent2num.pickle', 'rb'))
model = load(open('model.pickle','rb'))
label_dict = load(open('label_dict.pickle', 'rb'))
num_dict = load(open('num_dict.pickle', 'rb'))

temp_txt = u'国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
temp_txt = list(temp_txt)


# In[18]:


temp_num = sent2num(temp_txt)
print(temp_num[:5])


# In[24]:


# 根据输入得到标注推断
def predict_num(input_num,input_txt,
                model = model,
                label_dict=label_dict,
                num_dict=num_dict):
    input_num = np.array(input_num)
    predict_prob = model.predict_proba(input_num)
    predict_lable = model.predict_classes(input_num)
    for i , lable in enumerate(predict_lable[:-1]):
        # 如果是首字 ，不可为E, M
        if i==0:
            predict_prob[i, label_dict[u'E']] = 0
            predict_prob[i, label_dict[u'M']] = 0
        # 前字为B，后字不可为B,S
        if lable == label_dict[u'B']:
            predict_prob[i+1,label_dict[u'B']] = 0
            predict_prob[i+1,label_dict[u'S']] = 0
        # 前字为E，后字不可为M,E
        if lable == label_dict[u'E']:
            predict_prob[i+1,label_dict[u'M']] = 0
            predict_prob[i+1,label_dict[u'E']] = 0
        # 前字为M，后字不可为B,S
        if lable == label_dict[u'M']:
            predict_prob[i+1,label_dict[u'B']] = 0
            predict_prob[i+1,label_dict[u'S']] = 0
        # 前字为S，后字不可为M,E
        if lable == label_dict[u'S']:
            predict_prob[i+1,label_dict[u'M']] = 0
            predict_prob[i+1,label_dict[u'E']] = 0
        predict_lable[i+1] = predict_prob[i+1].argmax()
    predict_lable_new = [num_dict[x]  for x in predict_lable]
    result =  [w+'/' +l  for w, l in zip(input_txt,predict_lable_new)]
    return ' '.join(result) + '\n'


# In[26]:


temp = predict_num(temp_num,temp_txt)
print(temp)


def main():
    pass


if __name__ == '__main__':
    main()