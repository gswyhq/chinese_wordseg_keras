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
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Flatten ,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback


# - 步骤3：训练模型
train_label = load(open('train_label.pickle', 'rb'))
train_word_num = load(open('train_word_num.pickle','rb'))
init_weight_wv= load(open('init_weight.pickle','rb'))

nb_classes = len(np.unique(train_label))

# 建立两个字典
label_dict = dict(zip(np.unique(train_label), range(4)))
num_dict = {n:l  for l,n  in label_dict.items()}
print(label_dict)
print(num_dict)
# 将目标变量转为数字
train_label = [label_dict[y] for y in train_label]


# In[70]:


# 切分数据集
train_X, test_X, train_y, test_y = train_test_split(train_word_num, train_label , train_size=0.9, random_state=1)


# In[71]:


Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)


# In[72]:


print(len(train_X), 'train sequences')
print(len(test_X), 'test sequences')


# In[74]:


# 初始字向量格式准备
init_weight = [np.array(init_weight_wv)]


# In[79]:


batch_size = 256


# In[75]:


maxfeatures = init_weight[0].shape[0] # 词典大小


# In[87]:


# 一个普通的单隐层神经网络，输入层700，隐藏层100，输出层4
# 迭代时同时更新神经网络权重，以及词向量
print('Build model...')
model = Sequential()
# 词向量初始化，输入维度：词典大小，输出维度：词向量100
model.add(Embedding(maxfeatures, 100,weights=init_weight,input_length=7)) # 使用初使词向量可以增加准确率
# Embedding: 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]; 该层只能用作模型中的第一层。

model.add(Flatten())
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
# eg: (None, 64, 32, 32) --> (None, 65536)

model.add(Dense(100, input_dim=700, activation='relu'))
model.add(Dropout(0.5))  # 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。

# model.add(Activation('relu'))
model.add(Dense(nb_classes, input_dim=100, activation='softmax'))
# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# In[77]:


class EarlyStopping(Callback):
    def __init__(self, patience=0, verbose=0):
        super(Callback, self).__init__()

        self.patience = patience
        self.verbose = verbose
        self.best_val_loss = np.Inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        # if not self.params['do_validation']:
        #     warnings.warn("Early stopping requires validation data!", RuntimeWarning)

        cur_val_loss = logs.get('val_loss')
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1


# In[144]:


# train_X, test_X, Y_train, Y_test
print("Train...")
earlystop = EarlyStopping(patience=0, verbose=1)
result = model.fit(np.array(train_X), Y_train, batch_size=batch_size, nb_epoch=1,
          validation_split=0.1,callbacks=[earlystop])


# In[145]:


score = earlystop.model.evaluate(np.array(test_X), Y_test, batch_size=batch_size)
print('Test score:', score)


# In[146]:


# test数据集，准确率0.94
classes = earlystop.model.predict_classes(np.array(test_X), batch_size=batch_size)
# acc = np_utils.accuracy(classes, test_y) # 要用没有转换前的y
acc = sum([1 if k==v else 0 for k, v in zip(classes, test_y)])/len(test_y)
print('Test accuracy:', acc)


# In[2]:

dump(model, open('model.pickle', 'wb'))
# model = load(open('model.pickle','rb'))

dump(label_dict, open('label_dict.pickle', 'wb'))
dump(num_dict, open('num_dict.pickle', 'wb'))

def main():
    pass


if __name__ == '__main__':
    main()