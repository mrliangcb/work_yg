from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import word2vec


art=['我 篮球 哈哈 足球 可乐 学校 宿舍 摩托车','我 篮球 哈哈 足球 可乐 学校 宿舍 摩托车']


wv_model = Word2Vec(art, size=300, negative=5, workers=8, iter=5, window=3,
                        min_count=5)


print(wv_model)

a=[1,2,3,4,5,6]
b=[1,2,3,4,5,7]
for i,j in zip(a,b):
    print(i,j)


