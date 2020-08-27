import numpy as np 
import h5py
# x={'123':1,'456':2}
# x=np.array(x)
# np.savetxt('./test.txt', x, delimiter=',' ,encoding='utf-8',fmt = '%s')

x={'liang':1,'huang':2}
print(list(x))
print(list(x.values()))
# f = h5py.File(r"./mytestfile.hdf5", "w")
# dset = f.create_dataset("init", data=x)
# f.close

dic = dict(zip(list(x), list(x.values())))
print




# x=np.loadtxt('./test.txt' , delimiter=',' ,dtype='str')
# print(x)