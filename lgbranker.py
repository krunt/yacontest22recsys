#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.notebook import trange, tqdm
from scipy import sparse
import pandas as pd
import numpy as np
import gc
from collections import defaultdict, Counter
import math


# In[2]:


target_path = 'true_answers30.csv'
predict_path = 'submission_test30.csv'
predict_path90 = 'submission_test30_90.csv'
orig_path = 'test30.csv'

with open(target_path) as f:
    y_true = [int(x.strip()) for x in f.readlines()]


# In[3]:



with open(predict_path) as f:
    y_pred = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]

with open(predict_path90) as f:
    y_pred90 = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]
    


# In[4]:


with open(orig_path) as f:
    y_orig = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]


# In[5]:


tartists_pd = pd.read_csv('track_artists.csv')
art2track = defaultdict(list)
track2art = {}
for rec in tartists_pd.to_dict('records'):
    art2track[rec['artistId']].append(rec['trackId'])
    track2art[rec['trackId']] = rec['artistId']


# In[6]:


track_count = tartists_pd.shape[0]
track_hits_arr = np.zeros(track_count, dtype=np.int64)

total_list =  []

with open('train') as f:
    lines = f.readlines()
    for user_id, line in tqdm(enumerate(lines), total=len(lines)):
        user_track_pairs = list(map(lambda x: (user_id, int(x)), line.strip().split(' ')))
        total_list += user_track_pairs
        for track_id in list(map(int, line.strip().split(' '))):
            track_hits_arr[track_id] += 1


# In[7]:


# most_popular = np.argsort(-track_arr)[:1000]


# In[8]:


total_list = np.array(total_list, dtype=np.uint32)

track_cnt = tartists_pd.shape[0]
user_cnt = len(np.unique(total_list[:, 0]))

train_user_track_hits = sparse.coo_matrix(
    (np.repeat(1, total_list.shape[0]), (total_list[:, 0], total_list[:, 1])),
    shape=(user_cnt, track_cnt),
    dtype=np.float32,
).tocsc()


# In[ ]:





# In[9]:


def get_user_features(lst):
    ntracks = len(lst)
    
    artists = [track2art[track_id] for track_id in lst]
    c = Counter()
    for a in artists:
        c[a] += 1

    nartists = len(c)
    mean_acnt, std_acnt = np.mean(list(c.values())), np.std(list(c.values()))
    
    afill = [(c[art_id] / len(art2track[art_id])) for art_id in c.keys()]
    mean_afill, std_afill, max_afill = np.mean(afill), np.std(afill), np.max(afill)
    
    arepeats = 0
    adict = defaultdict(int)
    aprev = -1
    for a in artists:
        adict[a] += 1
        if adict[a] > 1 and aprev != a:
            arepeats += 1
        aprev = a
    
    return [ntracks, nartists, mean_acnt, std_acnt, mean_afill, std_afill, max_afill, arepeats]
    
def get_track_features(track_id):
    art_id = track2art[track_id]
    atrack_cnt = len(art2track[art_id])
    
    nhits = track_hits_arr[track_id]
    
    alist = sorted(art2track[art_id], key=lambda x: -track_hits_arr[x])
    popularity_index = alist.index(track_id) / atrack_cnt
    
    return [atrack_cnt, nhits, popularity_index]

def get_user_track_features(lst, qtrack_id):
    artists = [track2art[track_id] for track_id in lst]
    
    ret = [-1, -1, -1, -1, -1, -1]
    
#     try:
#         ret[0] = lst.index(qtrack_id) / len(lst)
#         ret[1] = lst[::-1].index(qtrack_id) / len(lst)
#     except ValueError:
#         pass
    
    inds = np.nonzero(np.array(lst) == qtrack_id)[0]
    if len(inds) > 0:
        ret[0] = inds[0] / len(lst)
        ret[1] = inds[-1] / len(lst)
    
#     try:
    art_id = track2art[qtrack_id]
#     print(artists, art_id)
    inds = np.nonzero(np.array(artists) == art_id)[0]
#     if ret[0] >= 0:
#         print(artists, art_id, inds)
    if len(inds) > 0:
        ret[2] = inds[0] / len(artists)
        ret[3] = inds[-1] / len(artists)
        ret[4] = len(inds) / len(lst)
        ret[5] = (inds[-1] - inds[0] + 1) / len(lst)
#     except ValueError:
#         pass
    
    return ret

def calc_mrr_pos(predict, answer):
#     for i in range(len(predict)):
#         if predict[i] == answer:
#             return i
#     return len(predict)

    try:
        return predict.index(answer)
    except ValueError:
        return len(predict)
    
def calc_mrr_pos_fast(predict_map, answer):
    if answer in predict_map:
        return predict_map[answer]
    return 100


# In[10]:


xlist = []
ylist = []
groups = []
tfcache = {}
for group_id, (orig, pred, pred90, answer) in tqdm(enumerate(zip(y_orig, y_pred, y_pred90, y_true)), total=len(y_pred)):
    uf = get_user_features(orig)
    
    tids = np.copy(pred)
    np.random.shuffle(tids)
    
    tlist = np.unique([answer,] + list(tids[:3]))
    for track_id in tlist:
        pos = calc_mrr_pos(pred, track_id)
        pos90 = calc_mrr_pos(pred90, track_id)
        if track_id not in tfcache:
            tfcache[track_id] = get_track_features(track_id)
        tf = tfcache[track_id]
        xlist.append(uf + tf + get_user_track_features(orig, track_id) + [pos, pos90]) # pos90, pos999])
        ylist.append(1 if track_id == answer else 0)
        
    groups.append(len(tlist))


# In[11]:


# [0]+list(tids[:3])


# In[12]:


xlist = np.array(xlist)
ylist = np.array(ylist)


# In[13]:


tlim = 25000
uidx = np.sum(groups[:tlim])
xtrainlist = xlist[:uidx]
ytrainlist = ylist[:uidx]
xtestlist = xlist[uidx:]
ytestlist = ylist[uidx:]
traingroups = groups[:tlim]
testgroups = groups[tlim:]


# In[14]:


from lightgbm.sklearn import LGBMRanker


# In[15]:




ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=30,
    importance_type='gain',
    verbose=1
)


# In[16]:


ranker = ranker.fit(
    xtrainlist,
    ytrainlist,
    group=traingroups,
    eval_set=[(xtestlist,ytestlist)],
    eval_group=[testgroups],
)


# In[ ]:





# In[ ]:


# In[19]:



predict_path = 'tot_submission.csv'
predict_path90 = 'tot_submission_90.csv'
orig_path = 'test'


with open(predict_path) as f:
    y_pred = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]

with open(predict_path90) as f:
    y_pred90 = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]
    
with open(orig_path) as f:
    y_orig = [[int(x) for x in line.strip().split(' ')] for line in f.readlines()]


# In[20]:


tfcache = {}

with open('tot_submission_lgb.csv', 'w') as fd:
    for group_id, (orig, pred, pred90) in tqdm(enumerate(zip(y_orig, y_pred, y_pred90)), total=len(y_pred)):
    
        uf = get_user_features(orig)
        
        pred_map = {item: i for i, item in enumerate(pred)}
        pred90_map = {item: i for i, item in enumerate(pred90)}
        
        xlist = []
        for track_id in pred:
            pos = calc_mrr_pos_fast(pred_map, track_id)
            pos90 = calc_mrr_pos_fast(pred90_map, track_id)
            if track_id not in tfcache:
                tfcache[track_id] = get_track_features(track_id)
            tf = tfcache[track_id]
            xlist.append(uf + tf + get_user_track_features(orig, track_id) + [pos, pos90]) # pos90, pos999])
        
        scores = ranker.predict(xlist)
        
        rpred = np.array(pred)[np.argsort(-scores)]
        
        fd.write(' '.join(map(str, np.array(rpred).flatten())) + '\n')


# In[ ]:




