#!/usr/bin/python
# coding: utf-8

# In[1]:


from tqdm import trange, tqdm
from scipy import sparse
import pandas as pd
import numpy as np
import gc
from collections import defaultdict
import math


# In[2]:


weigh_coeff, subsuffix = float(sys.argv[1]), (sys.argv[2] or "")


trackoccur_arr = np.load('tot_diag.npy')


# In[3]:


tartists_pd = pd.read_csv('track_artists.csv')
art2track = defaultdict(list)
track2art = {}
for rec in tartists_pd.to_dict('records'):
    art2track[rec['artistId']].append(rec['trackId'])
    track2art[rec['trackId']] = rec['artistId']


# In[4]:






# In[7]:


#weigh_coeff = 0.999
# weigh_coeff = 0.5
# weigh_coeff = 0.90
# weigh_coeff = 0.750


total_list =  []
total_weigh = []
y_orig = None
with open('test') as f:
    lines = f.readlines()
    y_orig = [[int(x) for x in line.strip().split(' ')] for line in lines]
    for user_id, line in tqdm(enumerate(lines), total=len(lines)):
        user_track_pairs = list(map(lambda x: (user_id, int(x)), line.strip().split(' ')))
        total_list += user_track_pairs
        total_weigh += list(weigh_coeff**np.arange(1,len(user_track_pairs)+1)[::-1])


# In[8]:


total_list = np.array(total_list, dtype=np.uint32)


# In[9]:


track_cnt = tartists_pd.shape[0]
user_cnt = len(np.unique(total_list[:, 0]))
(user_cnt, track_cnt)


# In[10]:


user_track_hits = sparse.coo_matrix(
    (total_weigh, (total_list[:, 0], total_list[:, 1])),
    shape=(user_cnt, track_cnt),
    dtype=np.float32,
).tocsr()


# In[11]:


nstep=100000
nparts=math.ceil(track_cnt/nstep)


# In[12]:


tids_tot = []
tvals_tot = []

for ipart in range(nparts):
    mat = sparse.load_npz('jac_tot_dz_%d.npz' % ipart).astype(np.float32)

    fr, to = ipart * nstep, min(ipart * nstep + nstep, track_cnt)

    ustep = 10000
    uparts=math.ceil(user_cnt / ustep)
    #uparts = 2

    utids_tot = []
    utvals_tot = []
    for upart in tqdm(range(uparts)):
        ufr, uto = upart * ustep, min(upart * ustep + ustep, user_cnt)

        umat = user_track_hits[ufr:uto, ...].dot(mat.transpose()).toarray()

        tids = np.argsort(-umat, axis=1)[:, :100]
        tvals = np.take_along_axis(umat, tids, axis=1)

        utids_tot.append(tids + fr)
        utvals_tot.append(tvals)

    utids_tot = np.vstack(utids_tot)
    utvals_tot = np.vstack(utvals_tot)
    
    tids_tot.append(utids_tot)
    tvals_tot.append(utvals_tot)
        


# In[13]:


d_tids_tot = np.hstack(tids_tot)
d_tvals_tot = np.hstack(tvals_tot)


# In[14]:


tids = np.argsort(-d_tvals_tot, axis=1)[:, :500]
tvals = np.take_along_axis(d_tvals_tot, tids, axis=1)
tids = np.take_along_axis(d_tids_tot, tids, axis=1)


# In[15]:

def filter_item(orig, pred):
    return np.array([x for x in pred if x not in orig])


with open('tot_submission%s.csv' % subsuffix, 'w') as fd:
    for orig, item in zip(y_orig, tids):
        item = filter_item(orig, item)[:100]
        fd.write(' '.join(map(str, np.array(item).flatten())) + '\n')


