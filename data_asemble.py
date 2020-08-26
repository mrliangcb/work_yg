
import h5py
import glob
import os
text_path=r'./src_split/text*'
label_path=r'./src_split/label*'
t_path_list=glob.glob(text_path)  #获得全名
l_path_list=glob.glob(label_path)  #获得全名
out_path=r'./src_assemble'


txt_all=[]
label_all=[]
print(t_path_list,l_path_list)


for i in t_path_list:
    with open(i, 'r',encoding='utf-8') as f:
        for row in f:
            row=repr(row)
            row = row.replace(r'\u200b', '')
            row=row.strip('\'')
            row=row.replace(r'\n', '')
            row=row.encode()
            txt_all.append(row)
for i in l_path_list:
    with open(i, 'r',encoding='utf-8') as f:
        for row in f:
            row=repr(row)
            row = row.replace(r'\u200b', '')
            row=row.strip('\'')
            row=row.replace(r'\n', '')
            row=row.encode()
            # print(row.decode('utf-8'))
            label_all.append(row)

print(len(txt_all))
print(len(label_all))
print(txt_all[:10])
print(label_all[:10])


out_txt=os.path.join(out_path, "text_all.h5")
f = h5py.File(out_txt,'w')
dset = f.create_dataset("text", data=txt_all)
f.close()


out_label=os.path.join(out_path, "label_all.h5")
f = h5py.File(out_label,'w')
dset = f.create_dataset("label", data=label_all)
f.close()



















