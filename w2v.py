import h5py
import jieba
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import word2vec
import time
print('1.44版本')


# 我要把这几个词加入到w2v训练语料中，然后训练，使得这个词有对应的w2v
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


txt_path=r'./src_assemble/text_all.h5'
lable_path=r'./src_assemble/label_all.h5'


print('正在解码txt')
f = h5py.File(txt_path, 'r')
print(list(f.keys()))
dset = f['text']
text_all=[]
print(len(dset))#679898
for i in range(len(dset)): 
    text=dset[i]
    text=text.decode()
    text_all.append(text)
f.close()

print('正在解码label')
f = h5py.File(lable_path, 'r')
print(list(f.keys()))
dset = f['label']
label_all=[]
print(len(dset))

for i in range(len(dset)):
    label=dset[i]
    label=label.decode()
    label_all.append(label)
f.close()

# print(text_all)
# print(label_all)


def filter_stopwords(words):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    return [word for word in words if word not in stop_words]


def participle_from_file(text_list):
    """加载数据文件，对文本进行分词"""
    data_list = []
    for line in text_list:
        # jieba.enable_parallel()

        words = jieba.cut(line.strip()) #输出的是对象

        word_list = list(words)

        # words = filter_stopwords(words)

        # jieba.disable_parallel()
        # data_list.append(' '.join(word_list) .strip()) 
        data_list.append(word_list) 

    return data_list

print('开始分词')
fenci_text=participle_from_file(text_all)   #就是   [['词1','词2'],['词3','词4']]
fenci_label=participle_from_file(label_all)

out_txt_path=r'./fenci/txt_fenci.txt'
with open(out_txt_path,'w', encoding='utf-8') as writer:
    for row in fenci_text:
        row=' '.join(row) .strip()
        writer.write(row+'\n')

out_label_path=r'./fenci/label_fenci.txt'
with open(out_label_path,'w', encoding='utf-8') as writer:
    for row in fenci_label:
        row=' '.join(row) .strip()
        writer.write(row+'\n')


# print(fenci_text)
# print(fenci_label)


# fenci_text=[ i.encode()  for i in fenci_text]
# fenci_label=[ i.encode()  for i in fenci_label]


# #(?# stop_w_path=r'./stopwords/哈工大停用词表.txt')

# 
# out_label_path=r'./fenci/label_spli.h5'

# f = h5py.File(out_txt_path, 'w')
# dset = f.create_dataset("text", data=fenci_text)
# f.close()

# f = h5py.File(out_label_path, 'w')
# dset = f.create_dataset("label", data=fenci_label)
# f.close()

# # 预料制作
# train_split=len(fenci_text)-500
# train_txt=fenci_text[:train_split]
# print('训练集长度:',len(train_txt)) #679398   少了500
# train_label=fenci_label[:train_split]

# train_w2v_list=train_txt+train_label

# for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
#     w=[[w]*10]
#     train_w2v_list=train_w2v_list+w #特殊词加入到预料


# print('训练语料:',train_w2v_list[-20:])
# print('训练语料:',len(train_w2v_list))


# print('start build w2v model')
# start_time=time.time()
# wv_model = Word2Vec(train_w2v_list, size=128, negative=5, workers=8, iter=50, window=3,
#                         min_count=5)
# finish_time=time.time()
# print('训练时间:',start_time-finish_time)
# # vocab = wv_model.wv.vocab

# dic={}
# for i,j in enumerate(wv_model.wv.index2word):
#     dic[j]=i
# print(dic)

# wv_model.save(r'./wv_model')
# # embedding_matrix = wv_model.wv.vectors
# # np.savetxt('./embedding_w.txt', embedding_matrix, fmt='%0.8f')









