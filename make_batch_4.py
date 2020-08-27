import os
import sys
import time
# import jieba
# import struct
import collections
import random
import torch
import numpy as np
from random import shuffle
from queue import Queue
from torch.autograd import Variable
# import h5py 
import config
import sys
from config import USE_CUDA, DEVICE
import random

print('make_batch_1.3版本')
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


def open_dic(path,mode):
    # print('open_dic的输入:',path)
    with open(path, 'r',encoding='utf-8') as f: #打开 (词 出现次数)词典
        out_dict={}
        for line in f:
            if line[0]==" ":
                out_dict[" "]=int(line[2:])
                print('遇到空符号了',line)
            else:
                pieces=line.split(' ')
                if mode=='w2i':
                    # print('{}_{}'.format(pieces[0],pieces[1]))
                    out_dict[pieces[0]]=int(pieces[1])
                else:
                    # print('i2w模式')
                    # print('{}_{}'.format(pieces[0],pieces[1]))
                    out_dict[int(pieces[0])]=pieces[1]

    return out_dict



# 读w2i i2w
class Vocab():
    def __init__(self,w2i_path,i2w_path):
        self.w2i_dic=open_dic(w2i_path,'w2i')
        
        self.i2w_dic=open_dic(i2w_path,'i2w')
        self._count=len(self.w2i_dic)
    def word2id(self,word):
        ''' 输入一个字，输出id '''
        if word not in self.w2i_dic:  #如果是不在id以内，则返回unk 的id
            return self.w2i_dic[UNKNOWN_TOKEN]
        return self.w2i_dic[word]
    def size(self):
        """获取加上特殊符号后的词汇表数量"""
        return self._count


class example: #处理好的 一个文章 以及他对应的摘要
    #包括pad这种  输入一个文章 [[词1],[词2]]  abs是一个摘要  w2i为一个字典   就是example
    # 首先是编码器输入
    """
    属性有：
    self.article_len  文章长度
    self.article_id 文章普通字典id
    self.article_oovs 每个句子有个oov词典
    self.article_id_extend_vocab  文章通过普通词典+oov词典编码   
    self.target  摘要从尾到头，max句子长度。再拼上结束符号


    """
    def __init__(self,article, abstract, vocab): # 含空格，非split的，输入一个句子
        article=article.strip()
        article_words = article.split() #不含空格 可能产生\u200b
        article_words = [t.strip(u'\u200b') for t in article_words]
        article_words = [t for t in article_words if t != ""]
        self.src_article=article_words
        
        # 1.处理文章
        if len(article_words) > config.max_enc_steps: #文章长度太大
            article_words = article_words[:config.max_enc_steps] #截断
        self.article_len = len(article_words) # 就是enc_len 最后文章字数
        self.article_id=[vocab.word2id(w) for w in article_words] #就是enc_input  含unk
        #2.处理摘要
        
        abstract=abstract.strip()
        abstract_words = abstract.split() # split 无空格
        abstract_words = [t.strip(u'\u200b') for t in abstract_words]
        abstract_words = [t for t in abstract_words if t != ""]

        self.src_abs=abstract_words
        abs_ids = [vocab.word2id(w) for w in abstract_words] #编码


        # 译码器输入
        start_decoding = vocab.word2id(START_DECODING) #开始的id
        stop_decoding = vocab.word2id(STOP_DECODING) # 结束的id
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding) # get_dec_inp_targ_seqs可以处理文章，也可以处理摘要
        self.dec_len = len(self.dec_input)

        if config.pointer_gen:
            #这个主要进一步处理 oov
            self.article_id_extend_vocab, self.article_oovs = self.article2ids(article_words, vocab) # id是每个词的id，以及有哪些oov词
            # article_words 就是split之后的文章  enc_input_extend_vocab装着下标
            abs_ids_extend_vocab = self.abstract2ids(abstract_words, vocab, self.article_oovs)
            # 目标编码和处理oov 
            # # target主要是  [句子，结束标号]
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        # 存储原始数据
        self.original_article = article
        self.original_abstract = abstract

    def pad_encoder_input(self,max_len, pad_id):
        while len(self.article_id) < max_len:
            self.article_id.append(pad_id) #尾巴添加pad，直到达到最大长度
        if config.pointer_gen:
            while len(self.article_id_extend_vocab) < max_len:
                self.article_id_extend_vocab.append(pad_id) #
                # article_id_extend_vocab 为扩展词典对应的 文章编码id

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

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
    #"""返回两个列表：将文章的词汇转换为id,包含oov词汇id; oov词汇"""

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

        # print('article_words:',article_words)
        # print('article2ids里面的的ids:',ids)

        # print('最大下标:',max(ids))
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

class make_batch_propose():
    '''
    enc_batch 为一个batch的文章 (含pad)
    enc_lens 一个batch文章的各个长度
    
    #输出经过处理的example batch
    '''
    def __init__(self,example_list,vocab, batch_size): #
        self.batch_size = batch_size
        example_list=self.make_sorted(example_list) #先排序 
        self.pad_id = vocab.word2id(PAD_TOKEN)
        self.init_encoder_seq(example_list) #这里对一个batch的example做填充  pad
        self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
        # self.store_orig_strings(example_list) # store the original strings
        
    def init_encoder_seq(self, example_list):
        # 
        max_enc_seq_len = max([ex.article_len for ex in example_list]) #遍历多个句子对象，获得最长的

        # Pad the encoder input sequences up to the length of the longest sequence 给encoder输入补0
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id) #逐个句子加pad

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32) #初始的enc_batch矩阵， bath大小）最长的句子长度  所以没有达到这个大小的，就设置为0
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.article_id[:] #已经pad了的
            self.enc_lens[i] = ex.article_len
            for j in range(ex.article_len):
                self.enc_padding_mask[i][j] = 1

        # pointer机制
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.article_id_extend_vocab[:]
            # print('batcher的enc_batch_extend_vocab字典:',self.enc_batch_extend_vocab)



    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:] #正常
            # print('self.target_batch是什么:',self.target_batch.shape,self.target_batch)
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    # def store_orig_strings(self, example_list):
    #     self.original_articles = [ex.original_article for ex in example_list] # list of lists
    #     self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    #     self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists

    def make_sorted(self,example_list):
        inputs=example_list
        sorted_list=[]
        inputs = sorted(inputs, key=lambda inp: inp.article_len, reverse=True)
        
        return inputs

    def get_input_from_batch(self):
        '''
        拿到一个batch的example_propose，输出训练数据
        '''
        atch_size = len(self.enc_lens)
        enc_batch = Variable(torch.from_numpy(self.enc_batch).long())
        enc_padding_mask = Variable(torch.from_numpy(self.enc_padding_mask)).float()
        enc_lens = self.enc_lens
        extra_zeros = None
        enc_batch_extend_vocab = None

        if config.pointer_gen:
            enc_batch_extend_vocab = Variable(torch.from_numpy(self.enc_batch_extend_vocab).long())
            # max_art_oovs is the max over all the article oov list in the batch
            if self.max_art_oovs > 0:
                extra_zeros = Variable(torch.zeros((self.batch_size, self.max_art_oovs)))
        c_t_1 = Variable(torch.zeros((self.batch_size, 2 * config.hidden_dim)))

        coverage = None
        if config.is_coverage:
            coverage = Variable(torch.zeros(enc_batch.size()))
        if USE_CUDA:
            enc_batch = enc_batch.to(DEVICE)

            enc_padding_mask = enc_padding_mask.to(DEVICE)

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.to(DEVICE)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.to(DEVICE)
        c_t_1 = c_t_1.to(DEVICE)

        if coverage is not None:
            coverage = coverage.to(DEVICE)

        return [enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage]

    def get_output_from_batch(self):
        dec_batch = Variable(torch.from_numpy(self.dec_batch).long())
        dec_padding_mask = Variable(torch.from_numpy(self.dec_padding_mask)).float()
        dec_lens = self.dec_lens
        max_dec_len = np.max(dec_lens)
        dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
        
        target_batch = Variable(torch.from_numpy(self.target_batch)).long()

        if USE_CUDA:
            dec_batch = dec_batch.to(DEVICE)
            dec_padding_mask = dec_padding_mask.to(DEVICE)
            dec_lens_var = dec_lens_var.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
        
        return [dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch]




def train_batch_out2(text_path,label_path,vocab,batch_size):
    # with open(text_path, 'r',encoding='utf-8') as f1:
    #     row_num=0
    #     for i in f1:
    #         row_num+=1
    example_list=[]
    text_count=0
    label_count=0
    now_step=0
    with open(text_path, 'r',encoding='utf-8') as f1:
        for text in f1:
            text_count+=1
    with open(label_path, 'r',encoding='utf-8') as f2:
        for text in f2:
            label_count+=1
    if text_count!=label_count:
        print('文章和摘要数量不同，出错!!')
        return
    else:
        print('数据集总长',text_count)
        batch_all_num=text_count/config.batch_size
        print('一共多少个batch:',batch_all_num)

    with open(text_path, 'r',encoding='utf-8') as f1:
        with open(label_path, 'r',encoding='utf-8') as f2:
            for text in f1:
                label=f2.readline()
                ex_batch=example(text,label,vocab)
                example_list.append(ex_batch)
                if len(example_list)>=batch_size:
                    out_batch=make_batch_propose(example_list,vocab,batch_size)#初始化batch
                    batch_input=out_batch.get_input_from_batch()
                    batch_output=out_batch.get_output_from_batch()
                    yield batch_input,batch_output,example_list
                    example_list=[]
                    

def train_batch_out(text_path,label_path,vocab,batch_size):
    # with open(text_path, 'r',encoding='utf-8') as f1:
    #     row_num=0
    #     for i in f1:
    #         row_num+=1
    example_list=[]
    text_count=0
    label_count=0
    now_step=0
    with open(text_path, 'r',encoding='utf-8') as f1:
        for text in f1:
            text_count+=1
    with open(label_path, 'r',encoding='utf-8') as f2:
        for text in f2:
            label_count+=1

    if text_count!=label_count:
        print('文章和摘要数量不同，出错!!')
        return
    else:
        print('数据集总长',text_count)
        batch_all_num=text_count/config.batch_size
        print('一共多少个batch:',batch_all_num)

    with open(text_path, ) as f1:
        text_all=f1.readlines()
        # random.shuffle(text_all)
        # random.shuffle(text_all)
        # text_all=text_all[:33]
    with open(label_path, 'r',encoding='utf-8') as f2:
        label_all=f2.readlines()
        # random.shuffle(label_all)
        # random.shuffle(label_all)
        # label_all=label_all[:33]
    print('前五个文章:',text_all[:5])
    print('前五个label:',label_all[:5])
    num_list=list(range(0,len(label_all)))
    random.shuffle(num_list) #做乱序的下标
    for i in num_list:
        ex_batch=example(text_all[i],label_all[i],vocab)
        example_list.append(ex_batch)

        if len(example_list)>=batch_size:
            out_batch=make_batch_propose(example_list,vocab,batch_size)#初始化batch
            batch_input=out_batch.get_input_from_batch()
            batch_output=out_batch.get_output_from_batch()
            yield batch_input,batch_output,example_list
            example_list=[]









# def main():
#     '''
#     用法:
#     对每个句子建立example对象
#     将example包成list，输入batch，获得batch的训练数据
#     '''
#     pass

# DATA_ROOT = "./weibo"
# train_w2i_path = os.path.join(DATA_ROOT, "train_w2i.txt")
# train_i2w_path = os.path.join(DATA_ROOT, "train_i2w.txt")
# train_text_path = os.path.join(DATA_ROOT, "train_text.txt")
# train_label_path = os.path.join(DATA_ROOT, "train_label.txt")
# vocab=Vocab(train_w2i_path,train_i2w_path)

# ite=0
# batch_gen=batch_out(train_text_path,train_label_path,vocab,4)
# for x,y in batch_gen:
#     print(ite)
#     ite+=1


# x='徐州 女孩 打游戏'
# y='徐州'
# ex1=example(x,y,vocab)
# x='老头 游戏 喝酒 篮球 公司'
# y='游戏'
# ex2=example(x,y,vocab)
# batch=make_batch_propose([ex1,ex2],vocab,2)
# print(batch.enc_batch) # 
# print(batch.enc_lens)


