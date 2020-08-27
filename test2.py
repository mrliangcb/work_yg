


import os
import collections
import torch
# train_text_path = r'D:\coder\houcang\NLP\NLP\6-2课堂代码\PGN_GITHU_zingp\PGN_newGPU\weibo\train_art_sum_prep.txt'
# with open(train_text_path, 'r', encoding='utf-8') as f_art:
#     for i in f_art:
#         art_tokens = i.split(' ')
#         print(art_tokens)

# vocab_counter = collections.Counter()
# vocab_counter.update(['words'])
# vocab_counter.update('a')
# vocab_counter.update(['a','b'])
# pirnt()
# print(vocab_counter)



# x=torch.ones(3,3).cuda(0)
# print(x+1)


# x=[1,2,3,4,5,6,7,8,9,0]
# def gen(x):
#     for j in range(len(x)):
#         yield x[j]
#         if j>=6:
#             return 

# ex=gen(x)
# for i in ex:
#     print(i)
#     if i ==None:
#         print('None了')
# import torch
# import numpy as np
# # x=torch.ones(3,3)
# # y=torch.Tensor([0])
# # print(x/torch.log(y))

# xx=np.nan
# x=torch.Tensor([[1,2,3],[xx,6,7]])
# print(torch.isnan(x))




# y=['\u200b','']

# z=[i.strip('\u200b') for i in y]
# z=[i.strip('') for i in z]
# print(z)
import torch
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)







