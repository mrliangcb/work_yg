import h5py








txt_path=r'./src_assemble/text_all.h5'
lable_path=r'./src_assemble/label_all.h5'



f = h5py.File(txt_path, 'r')
print(list(f.keys()))
dset = f['text']
text_all=[]
print(len(dset))
for i in range(10):
    text=dset[i]
    text=text.decode()
    text_all.append(text)
f.close()


f = h5py.File(lable_path, 'r')
print(list(f.keys()))
dset = f['label']
label_all=[]
print(len(dset))
for i in range(10):
    label=dset[i]
    label=label.decode()
    label_all.append(label)
f.close()

print(text_all)
print(label_all)


def participle_from_file(text_list):
    """加载数据文件，对文本进行分词"""
    data_list = []
    for line in text_list:
        # jieba.enable_parallel()
        words = jieba.cut(line.strip())
        word_list = list(words)
        # jieba.disable_parallel()
        data_list.append(' '.join(word_list) .strip()) 
    return data_list

fenci_text=participle_from_file(text_all)
fenci_label=participle_from_file(label_all)

print(fenci_text)
print(fenci_label)















