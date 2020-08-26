import h5py








txt_path=r'./src_assemble/text_all.h5'




f = h5py.File(txt_path, 'r')
print(list(f.keys()))
dset = f['text']
text_all=[]
print(len(dset))
for i in range(10):
    text=dset[i]
    text=text.decode()
    text_all.append(text)
