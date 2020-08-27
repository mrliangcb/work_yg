import os
import sys
import time
import jieba
import struct
import collections
import random
import torch
import numpy as np
from random import shuffle
from queue import Queue
from torch.autograd import Variable

# import config
import sys

print('make_vocab的1.3版本')
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


def save_dic(dic,save_path):
    keys=dic.keys()
    values=dic.values()
    with open(save_path,'w', encoding='utf-8') as writer:
        for i,j in zip(keys,values):
            writer.write('{} {}\n'.format(i,j)) #解码的时候要转回int





def make_train_vocab(words_count_path,w2i_path,i2w_path,max_size): # 
    word2id_dic = {}
    id2word_dic = {}
    count = 0 #现在处理第几个单词

    #先做四个词和ID
    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
        word2id_dic[w] = count #count是计数有多少个词收揽进来了
        id2word_dic[count] = w
        count += 1
    
    with open(words_count_path, 'r',encoding='utf-8') as words_count: #打开 (词 出现次数)词典
        for line in words_count:
            pieces = line.split()
            if len(pieces) != 2:
                print('词典(词 出现次数) 不是两项')
            word_zh = pieces[0]
            if word_zh in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                print('词典1出现特殊符号，重来')
                return 
            if word_zh in word2id_dic: # 若中文在词典key中了
                print('词典中已经有这个词',word_zh)
            word2id_dic[word_zh] = count
            id2word_dic[count]=word_zh
            count+=1
            if max_size != 0 and self._count >= max_size:
                print("完成词典2，共 {} 个词. 最新词: {}".format(count, id2word_dic[count-1]))
                break
    # save_dic(word2id_dic,w2i_path)
    # save_dic(id2word_dic,i2w_path)
    # print("完成词典2，共 {} 个词. 最新词: {}".format(count, id2word_dic[count-1]))


# DATA_ROOT = "./weibo"
# train_w2i_path = os.path.join(DATA_ROOT, "train_w2i.txt")
# train_i2w_path = os.path.join(DATA_ROOT, "train_i2w.txt")
# words_count_path=os.path.join(DATA_ROOT, "words_count.txt")
# make_train_vocab(words_count_path,train_w2i_path,train_i2w_path,config.vocab_size) # 词 对应编码 字典

from gensim.models.word2vec import LineSentence, Word2Vec
wv_model = Word2Vec.load('./wv_model')

print('制作词典word2id')
w2i={}
i2w={}
for i,j in enumerate(wv_model.wv.index2word):
    w2i[j]=i
    i2w[i]=j

print('pad的id',dic['[PAD]'])
print(dic['[UNK]'])
dic_out_path1=r'./word2id_dic.txt'
dic_out_path2=r'./id2word_dic.txt'

save_dic(w2i,dic_out_path1)
save_dic(i2w,dic_out_path2)



