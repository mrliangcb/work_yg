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

# 路径设置
DATA_ROOT = "..\weibo" #数据集文件夹
FINISHED_FILE_DIR = os.path.join(DATA_ROOT, "finished_files")


##############################   公共部分    #############################
def timer(func):
    """耗时装饰器，计算函数运行时长"""
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs) #执行带@的函数
        end = time.time()
        cost = end - start
        print(f"Cost time: {cost} s")
        return r
    return wrapper

def save_dic(dic,path):
    f = h5py.File(path, "w")
    dset1 = f.create_dataset("keys", data=list(dic))#init主键
    dset1 = f.create_dataset("values", data=list(dic.values()))
    f.close()

def open_dic(path):
      f = h5py.File(path, 'r')
      keys = f['keys']
      values=f['values']
      out=dict(zip(keys,values))
      return out

###############################    做词典   ###############################
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
# 词汇表大小
VOCAB_SIZE = 50_000  


def make_vocab1(article,abstract,vocab1_path): # 输入非split
    vocab_counter = collections.Counter()
    art_tokens = article.split(' ')
    abs_tokens = abstract.split(' ') # 不含 <s>  </s>  字典1没有这个的
    global_tokens=art_tokens+abs_tokens
    tokens = [t.strip() for t in tokens]     # 对每个token去掉开头结尾的空字符
    tokens = [t for t in tokens if t != ""]  # 删除空行
    vocab_counter.update(tokens)
    vocab_counter=vocab_counter.most_common(VOCAB_SIZE) #取出最常出现的部分词典
    with open(vocab1_path,'w', encoding='utf-8') as writer:      
        for word, count in vocab_counter:
            writer.write(word + ' ' + str(count) + '\n') # 中文词 出现次数
    print('写完字典1')



class vocab():
    def __init__(self,w2i_path,i2w_path):
        '''  加载已有的字典  '''
        self.w2i_dic=open_dic(w2i_path)
        self.i2w_dic=open_dic(i2w_path)
    def word2id(word):
        ''' 输入一个字，输出id '''
        if word not in self.w2i_dic:  #如果是不在id以内，则返回unk 的id
            return self.w2i_dic[UNKNOWN_TOKEN]
        return self.w2i_dic[word]


###############################   预处理   ########################################

# 1.原数据预处理
#分词
@timer
def participle_from_file(filename): #传入地址全名 
    """加载数据文件，对文本进行分词"""
    data_list = []
    with open(filename, 'r', encoding= 'utf-8') as f:
        for line in f:  # 不用f.readline()   .readlines()
            jieba.enable_parallel()
            words = jieba.cut(line.strip()) #用jieba来切
            word_list = list(words)
            jieba.disable_parallel()
            data_list.append(' '.join(word_list).strip()) # str "词1 词2 词3"
    return data_list


# 2.训练数据预处理

class example: #处理好的 一个文章 以及他对应的摘要
    #包括pad这种  输入一个文章 [[词1],[词2]]  abs是一个摘要  w2i为一个字典   就是example
    # 首先是编码器输入
    def __init__(self,article, abstract, vocab): # 含空格，非split的，输入一个句子
        article_words = article.split() #不含空格
        # 1.处理文章
        if len(article_words) > config.max_enc_steps: #文章长度太大
            article_words = article_words[:config.max_enc_steps] #截断
        self.article_len = len(article_words) # 就是enc_len 最后文章字数
        self.article_id=[vocab.word2id(w) for w in article_words] #就是enc_input  含unk
        #2.处理摘要
        abstract_words = abstract.split() # split 无空格
        abs_ids = [vocab.word2id(w) for w in abstract_words] #编码

        # 译码器输入
        start_decoding = vocab.word2id(START_DECODING) #开始的id
        stop_decoding = vocab.word2id(STOP_DECODING) # 结束的id
        self.dec_input, self.target = get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding) # get_dec_inp_targ_seqs可以处理文章，也可以处理摘要
        self.dec_len = len(self.dec_input)

        if config.pointer_gen:
            #这个主要进一步处理 oov
            self.article_id_extend_vocab, self.article_oovs = self.article2ids(article_words, vocab) # id是每个词的id，以及有哪些oov词
            # article_words 就是split之后的文章  enc_input_extend_vocab装着下标
            abs_ids_extend_vocab = self.abstract2ids(abstract_words, vocab, self.article_oovs)
            # 目标编码和处理oov 
            # # target主要是  [句子，结束标号]
            _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)


    def get_dec_inp_targ_seqs(self, abs_id, max_len, start_id, stop_id):
        inp = [start_id] + abs_id[:]  #教师模式，就是将摘要答案拼上开头结尾，然后输入decoder
        target = abs_id[:]
        if len(inp) > max_len:  # 截断
            inp = inp[:max_len]
            target = target[:max_len] # 没有结束标志
        else:   # 无截断
            target.append(stop_id)    # 结束标志
        assert len(inp) == len(target)
        return inp, target

    def article2ids(self,article_words, vocab):
    """返回两个列表：将文章的词汇转换为id,包含oov词汇id; oov词汇"""
        ids = []
        oovs = []
        unk_id = vocab.word2id(UNKNOWN_TOKEN)
        for w in article_words:
            i = vocab.word2id(w)
            if i == unk_id:         # If w is OOV   若这个词不经常出现，被字典截去了
                if w not in oovs:   # Add to list of OOVs  
                    oovs.append(w) #若这个词不在oovs里面，就加进去

                    oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
                    # 并且得到这个oov词的下标
                    ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
                    # ids 装着一对数字，显示OOV集合中，下一个词的开头
            else:
                ids.append(i)
        return ids, oovs #返回目前oov集合的新位置，还有当前oovs集合  一个文章对应一个oov集合

    def abstract2ids(self,abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i) #当前摘要词不在文章OOV集合中
    return ids


class make_batch_gen():
    def __init__(self,example_list_gen,vocab, batch_size): #
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)
        self.init_encoder_seq(example_list) #这里对一个batch的example做填充  pad


    def init_encoder_seq(self, example_list):
        # 
        max_enc_seq_len = max([ex.enc_len for ex in example_list]) #遍历多个句子对象，获得最长的

        # Pad the encoder input sequences up to the length of the longest sequence 给encoder输入补0
        for ex in example_list:
            self.pad_encoder_input(ex,max_enc_seq_len, self.pad_id) #逐个句子加pad

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32) #初始的enc_batch矩阵， bath大小）最长的句子长度  所以没有达到这个大小的，就设置为0

        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

    def pad_encoder_input(self, ex,max_len, pad_id):
        while len(ex.enc_input) < max_len:
            ex.enc_input.append(pad_id) #尾巴添加pad，直到达到最大长度
        if config.pointer_gen:
            while len(ex.article_id_extend_vocab) < max_len:
                self.article_id_extend_vocab.append(pad_id) #
                # article_id_extend_vocab 为扩展词典对应的 文章编码id




# 主函数
def main():
    src_train_file = os.path.join(DATA_ROOT, "train_text.txt") #原文章
    src_label_file = os.path.join(DATA_ROOT, "train_label.txt") #原摘要
    article_str = participle_from_file(src_train_file) # 切词 大概耗时10分钟
    abstract_str = participle_from_file(src_label_file) # 输出非split
    #保存切词结果list(str())!!!!

    #划分训练接测试集  训练集有训练集的字典，测试集有测试集的字典!!!!
    # 训练集多少
    all_data_len=len(article_str)
    train_num=70%
    article_str_train=article_str[:all_data_len*train_num] 
    abstract_str_train=abstract_str[:all_data_len*train_num]

    vocab1_path=os.path.join(FINISHED_FILE_DIR,"vocab")
    make_vocab1(article_str_train,abstract_str_train,vocab1_path) #生成 词 出现次数
    w2i_path=r"./word2id_dic.h5py"
    i2w_path=r"./id2word_dic.h5py"。、
    make_vocab2(vocab1_path,w2i_path,i2w_path) # 词 对应编码 字典

    #读取字典，  建立vocab对象，读取两个字典 w2i i2w


    



    #处理每个句子












