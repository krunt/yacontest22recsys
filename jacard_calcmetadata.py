#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm.notebook import trange, tqdm
from scipy import sparse
import pandas as pd
import numpy as np
import gc
import math


# In[ ]:


nparts = 5
nstep=100000


# In[ ]:


diagarr = []
for ipart in tqdm(range(nparts)):
#     mat = sparse.load_npz('mat_%d.npz' % ipart).astype(np.float32) # .tocsc()
    mat = sparse.load_npz('tot_mat_%d.npz' % ipart).astype(np.float32) # .tocsc()
    smat = mat[:, ipart*nstep:]
    diagarr.append(smat.diagonal())


# In[ ]:


diagarr = np.concatenate(diagarr)


# In[ ]:


# with open('diag.npy', 'wb') as fd:
with open('tot_diag.npy', 'wb') as fd:
    np.save(fd, diagarr)


# In[ ]:


nstep=100000
tartists_pd = pd.read_csv('track_artists.csv')
track_cnt = tartists_pd.shape[0]
nparts=math.ceil(track_cnt/nstep)


# trackoccur_arr = np.load('diag.npy')
trackoccur_arr = np.load('tot_diag.npy')


# In[ ]:





# In[ ]:


def apply_jacard(mat, fr, to):
    lim = 60000
    smat = mat[:lim, ...]
    smat_shape = smat.shape
#     print('A')
    rows, cols, vals = sparse.find(smat)
#     print('B')
    del smat
    vals = vals / (trackoccur_arr[rows+fr] + trackoccur_arr[cols] - vals + 1e-9)
    
    gc.collect()
    
#     print('C')
    
    mat0 = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=smat_shape,
        dtype=np.float32).tocsr()
    
    del rows, cols, vals
    gc.collect()
    
#     print('D')
    
    smat = mat[lim:, ...]
    smat_shape = smat.shape
#     print('E')
    rows, cols, vals = sparse.find(smat)
#     print('G')
    del smat
    vals = vals / (trackoccur_arr[rows+fr+lim] + trackoccur_arr[cols] - vals + 1e-9)
#     vals[(rows+fr+lim) == cols] = 0
    
    gc.collect()
    
#     print('Z')
    
    mat1 = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=smat_shape,
        dtype=np.float32).tocsr()
    
    del rows, cols, vals
    gc.collect()
    
#     print('Y')
    
    ret = sparse.vstack([mat0, mat1])
    for i in tqdm(range(mat.shape[0])):
        ret[i, i + fr] = 0
        
    return ret


# In[ ]:




for ipart in tqdm(range(nparts)):
    mat = sparse.load_npz('tot_mat_%d.npz' % ipart).astype(np.float32).tocsr()

    fr, to = ipart * nstep, min(ipart * nstep + nstep, track_cnt)
    
    mat = apply_jacard(mat, fr, to)
    
    sparse.save_npz('jac_tot_dz_%d.npz' % ipart, mat)
    


# In[ ]:


# mat = sparse.load_npz('mat_%d.npz' % 0).astype(np.float32).tocsr()


# In[ ]:





# In[ ]:


# dmat = apply_jacard(mat, 0, 100000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




