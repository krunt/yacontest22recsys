#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.notebook import trange, tqdm
from scipy import sparse
import pandas as pd
import numpy as np
import gc
import math


# In[2]:


total_list =  []
with open('train') as f:
    lines = f.readlines()
    for user_id, line in tqdm(enumerate(lines), total=len(lines)):
        user_track_pairs = list(map(lambda x: (user_id, int(x)), line.strip().split(' ')))
        total_list += user_track_pairs


# In[3]:


total_list = np.array(total_list, dtype=np.uint32)


# In[4]:


gc.collect()


# In[5]:


tartists_pd = pd.read_csv('track_artists.csv')


# In[6]:


track_cnt = tartists_pd.shape[0]
user_cnt = len(np.unique(total_list[:, 0]))
(user_cnt, track_cnt)


# In[ ]:





# In[7]:


user_track_hits = sparse.coo_matrix(
    (np.repeat(1, len(total_list)), (total_list[:, 0], total_list[:, 1])),
    shape=(user_cnt, track_cnt),
    dtype=np.uint32,
).tocsr()


# In[8]:


# track_cooccurrence = user_track_hits.transpose().dot(user_track_hits)


# In[9]:


# %%time 

# mat = user_track_hits[:, :10000].transpose().dot(user_track_hits)


# In[11]:


nstep=100000
nparts=math.ceil(track_cnt/nstep)
for ix, i in tqdm(enumerate(range(0, track_cnt, nstep)), total=nparts):
    fr, to = i, min(i + nstep, track_cnt)
#     print(fr, to)
    mat = user_track_hits[:, fr:to].transpose().dot(user_track_hits)
    sparse.save_npz('tot_mat_%d.npz' % ix, mat)


# In[ ]:





# In[ ]:




