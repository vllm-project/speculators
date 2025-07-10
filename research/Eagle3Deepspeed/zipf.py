import torch 
import os
import numpy as np 
allIds=np.zeros(128256)
total=0


n=5000

prefix="/ibmnetwork/FF/sharegpt/0"
files=os.listdir(prefix)

for i in range(0,(n)):
    print(files[i])
    data=torch.load(prefix+'/'+files[i])

    ids=(data['input_ids'].numpy())
    del data
    unique_values, counts = np.unique(ids, return_counts=True)
    allIds[unique_values]+=counts
    total+=len(ids)

prefix="/ibmnetwork/FF/ultrachat_sft/0"
files=os.listdir(prefix)

for i in range(0,n):
    print(files[i])
    data=torch.load(prefix+'/'+files[i])
    ids=(data['input_ids'].numpy())
    del data
    unique_values, counts = np.unique(ids, return_counts=True)
    allIds[unique_values]+=counts
    total+=len(ids)

vocab_size=32000

keep=np.argsort(-1*allIds )[:vocab_size]


# vocab_size=np.count_nonzero(allIds)




print(vocab_size)
t2d=np.full(128256, False, dtype=bool)
t2d[keep]=True
print(np.sum(t2d))


np.save("t2d.npy", t2d)



d2t=(np.arange(0,len(allIds))[t2d])-np.arange(0, vocab_size)
print(d2t)
np.save("d2t.npy", d2t)