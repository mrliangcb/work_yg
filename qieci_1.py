import os
import sys
import time
import jieba

import struct
import collections



def timer(func):
    """耗时装饰器，计算函数运行时长"""
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        print(f"Cost time: {cost} s")
        return r
    return wrapper
 
@timer
def participle_from_file(filename):
    """加载数据文件，对文本进行分词"""
    data_list = []
    with open(filename, 'r', encoding= 'utf-8') as f:
        for line in f:
            # jieba.enable_parallel()
            words = jieba.cut(line.strip())
            word_list = list(words)
            # jieba.disable_parallel()
            data_list.append(' '.join(word_list).strip())
    return data_list

def save_file(filename, li):
    """预处理后的数据保存到文件"""
    with open(filename, 'w+', encoding='utf-8') as f:
        for item in li:
            f.write(item + '\n')
    print(f"Save {filename} ok.")


def build_train_val(article_data, summary_data, train_num=600_000): #只是划分，没有处理句子
    """划分训练和验证数据"""
    train_text = []
    train_label=[]
    val_text = []
    val_label=[]
    n = 0
    for text, summ in zip(article_data, summary_data):
        n += 1
        if n <= train_num:
            train_text.append(text)
            train_label.append(summ)
        else:
            val_text.append(text)
            val_label.append(summ)
    return train_text, train_label,val_text,val_label



DATA_ROOT = "./weibo"
def main():
    # os.mkdir(DATA_ROOT) 
    src_train_file = os.path.join(DATA_ROOT, "train_text_src.txt") #原文章
    src_label_file = os.path.join(DATA_ROOT, "train_label_src.txt") #原摘要

    train_text_path = os.path.join(DATA_ROOT, "train_text.txt")# output
    train_label_path = os.path.join(DATA_ROOT, "train_label.txt")# output
    val_text_path = os.path.join(DATA_ROOT, "val_text.txt")# output
    val_label_path = os.path.join(DATA_ROOT, "val_label.txt")# output

    article_data = participle_from_file(src_train_file)     # 大概耗时10分钟
    summary_data = participle_from_file(src_label_file)

    train_split = 600_000
    train_text, train_label,val_text,val_label = build_train_val(article_data, summary_data, train_num=train_split)
    
    save_file(train_text_path, train_text)
    save_file(train_label_path, train_label)
    save_file(val_text_path, val_text)
    save_file(val_label_path, val_label)

    print('切词完成')

main()















