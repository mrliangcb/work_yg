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
import h5py
import config
import sys








def words_count(article_path,abstract_path,save_path): # 输入非split
    vocab_counter = collections.Counter()
    with open(article_path, 'r', encoding='utf-8-sig') as f_art:
        with open(abstract_path, 'r', encoding='utf-8-sig') as f_abs:
            for article in f_art:
                abstract=f_abs.readline()

                art_tokens = article.split(' ')
                art_tokens = [t.strip() for t in art_tokens]
                art_tokens = [t.strip(u'\u200b') for t in art_tokens] #去掉\u的元素
                art_tokens=[i for i in art_tokens if len(i)>0] #删去空元素
                # .encode('utf-8').decode('utf-8-sig').strip()
                # art_tokens=[t.encode('utf-8').decode('utf-8-sig').strip() for t in art_tokens]

                abs_tokens = abstract.split(' ') # 有\n  \u2006b结尾
                abs_tokens = [t.strip() for t in abs_tokens] # 去不掉\u2006
                abs_tokens = [t.strip(u'\u200b') for t in abs_tokens]
                # abs_tokens=[i for i in abs_tokens if len(i)>0] #删去空元素
                abs_tokens = [t for t in abs_tokens if t != ""]
                # abs_tokens=[t.encode('utf-8').decode('utf-8-sig').strip() for t in abs_tokens]

                global_tokens=art_tokens+abs_tokens
                vocab_counter.update(global_tokens) #统计这句字数

    vocab_counter=vocab_counter.most_common(VOCAB_SIZE) #取出最常出现的部分词典，most之前是一个dict，most之后就是一个list
    with open(save_path,'w', encoding='utf-8') as writer:      
        for word, count in vocab_counter:
            writer.write(word + ' ' + str(count) + '\n') # 中文词 出现次数
    print('写完字典1')

VOCAB_SIZE = 50_000  
DATA_ROOT = "./weibo"
train_text_path = os.path.join(DATA_ROOT, "train_text.txt")# output
train_label_path = os.path.join(DATA_ROOT, "train_label.txt")# output
val_text_path = os.path.join(DATA_ROOT, "val_text.txt")# output
val_label_path = os.path.join(DATA_ROOT, "val_label.txt")# output

words_count_save_path=os.path.join(DATA_ROOT, "words_count.txt")

words_count(train_text_path,train_label_path,words_count_save_path)
































