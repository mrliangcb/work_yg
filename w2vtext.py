from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import word2vec


art=[['我','篮球','哈哈','足球','可乐']]


wv_model = Word2Vec(art, size=300, negative=5, workers=8, iter=5, window=3,
                        min_count=1)


print(wv_model)

print('第一个:',wv_model.wv.index2word)  #(1,300)
print(wv_model.wv.vocab)
for i in wv_model.wv.index2word:
    print('这个i',i)
# wv_model.wv.index2word
# vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
# model.wv['sky']  表示输出sky这个词的特征映射结果

a=['1,2,3,4']
b=['11,13,5476']
print(a+b)

a=[1,2,3,4,5,6]
b=[1,2,3,4,5,7]
for i,j in zip(a,b):
    print(i,j)


