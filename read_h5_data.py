

import h5py
txt_path=r'./src_assemble/text_all.txt'


f = h5py.File(txt_path, 'r')
print(list(f.keys()))
dset = f['text']
print(len(dset))
for i in range(10):
    text=dset[i]
    text=text.decode()
    print(repr(text))





























