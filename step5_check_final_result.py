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

# 步骤5：检查最终的效果

import codecs
import sys
from os import path
from pickle import dump,load

from step4_use_cnn_model_word_segmentation import predict_num

sent2num = load(open('sent2num.pickle', 'rb'))
test_file = './data/icwb2-data/testing/msr_test.utf8'
with open(test_file,'r') as f:
    lines = f.readlines()
    test_texts = [list(line.strip()) for line in lines]


# In[ ]:


test_output = []
for line in test_texts:
    test_num = sent2num(line)
    output_line = predict_num(test_num,input_txt=line)
    test_output.append(output_line)


# In[31]:


with open('./data/icwb2-data/testing/msr_test_output.utf8','w') as f:
    f.writelines(test_output)


# In[32]:


input_file = './data/icwb2-data/testing/msr_test_output.utf8'
output_file = './data/icwb2-data/testing/msr_test.split.tag2word.utf8'


# In[33]:



def character_2_word(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    # 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
    for line in input_data.readlines():
        char_tag_list = line.strip().split()
        for char_tag in char_tag_list:
            char_tag_pair = char_tag.split('/')
            char = char_tag_pair[0]
            tag = char_tag_pair[1]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else: # tag == 'S'
                output_data.write(' ' + char + ' ')
        output_data.write("\n")
    input_data.close()
    output_data.close()

character_2_word(input_file, output_file)


# - 最终使用perl脚本检验的F值为0.913

# command = '''./data/icwb2-data/scripts/score data/icwb2-data/gold/msr_training_words.utf8 data/icwb2-data/gold/msr_test_gold.utf8 data/icwb2-data/testing/msr_test.split.tag2word.utf8 > deep.score'''
# os.system(command=command)



def main():
    pass


if __name__ == '__main__':
    main()