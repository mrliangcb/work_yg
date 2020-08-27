import numpy as np 
import h5py
import re
# x={'123':1,'456':2}
# x=np.array(x)
# np.savetxt('./test.txt', x, delimiter=',' ,encoding='utf-8',fmt = '%s')

# x={'liang':1,'huang':2}
# print(list(x))
# print(list(x.values()))
# # f = h5py.File(r"./mytestfile.hdf5", "w")
# # dset = f.create_dataset("init", data=x)
# # f.close

# dic = dict(zip(list(x), list(x.values())))
# print




# x=np.loadtxt('./test.txt' , delimiter=',' ,dtype='str')
# print(x)

# def clean_sentence(sentence):
#     '''
#     特殊符号去除
#     :param sentence: 待处理的字符串
#     :return: 过滤特殊字符后的字符串
#     '''
#     if isinstance(sentence, str):
#         # return re.sub(
#         #     r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
#         #     '', sentence)
#         return re.sub(
#             r'http\:\/\/.*?\s',
#             '', sentence)

#     else:
#         return ' '

# x='我们http://www.baidu.com/ ，@hello企业 http://www.baidu.com'
# y=clean_sentence(x)
# print(y)
from gensim.models.word2vec import LineSentence, Word2Vec

wv_model = Word2Vec.load('./wv_model')

embedding_matrix = wv_model.wv.vectors
print(embedding_matrix)
print(type(embedding_matrix))
print(len(embedding_matrix))



