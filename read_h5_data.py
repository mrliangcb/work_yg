

import h5py
path=r'E:\coder\U_code\NLP\MY_PGN\weibo\src_assemble\text_all.h5'


f = h5py.File(path, 'r')
print(list(f.keys()))
dset = f['text']
print(len(dset))
for i in range(10):
    text=dset[i]
    text=text.decode()
    print(repr(text))





























