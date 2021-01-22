#%%
%load_ext autoreload
%autoreload 2

import multiview_clustering as mc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import pandas as pd
from sklearn.cluster import SpectralClustering
#%%[markdown]
# ## Synthetic data creation

#%%
data_type = 'bool' # or linear

n_data = 350
n_dim = 4

S = np.zeros((n_dim,n_data,n_data))# Similarity tensor
cluster_members = [50,100,200]

# Contruct labels
y = []
for i,c in enumerate(cluster_members):
    y.extend(np.ones(c)*i) 

intra_c_prob = [[0.8,0.1,0.7],[0.05,0.4,0.06],[0.6,0.8,0.67],[0.6,0.4,0.1]]
noise_d = [0.15,0.05,0.3,0.14]
intra_cluster_proba = 0.5

for d in range(n_dim):
    i = 0
    for c in range(len(cluster_members)):
        if data_type == "bool" :
            S[d][i:i+cluster_members[c],i:i+cluster_members[c]] = np.random.binomial(1,intra_c_prob[d][c],size=cluster_members[c]**2).reshape((cluster_members[c],cluster_members[c])) 
        elif data_type == 'linear':
             S[d][i:i+cluster_members[c],i:i+cluster_members[c]] = np.random.normal(loc=intra_c_prob[d][c],scale=0.2,size=cluster_members[c]**2).reshape((cluster_members[c],cluster_members[c]))
        else :
            raise ValueError
        
        i = i + cluster_members[c]

    # Add noise
    if data_type == "bool" :
        S[d] = S[d] + np.random.binomial(1,noise_d[d],size=S[d].size).reshape(S[d].shape) #np.random.normal(loc=noise_d[d],scale=0.9,size=S[d].size).reshape(S[d].shape)
    elif data_type == 'linear':
        S[d] = S[d] + np.random.normal(loc=noise_d[d],scale=0.9,size=S[d].size).reshape(S[d].shape)
    else :
        raise ValueError

    # Make symmetric
    S[d] = (np.tril( S[d]) + np.triu(S[d].T, 1))

    S[d] = abs(S[d]) # remove negative
    S[d] = np.where(S[d]>=1,1,S[d]) # Remove val > 1

    # Set diag to single value
    di,dj = np.diag_indices_from(S[d])
    S[d][di,dj] = 1

# %%

fig,ax = plt.subplots(1,n_dim,figsize=(13,4))
for d in range(n_dim):
    ax[d].imshow(S[d])

fig.suptitle("Similarity Matrices")
plt.show()

#%%

shuffled = np.arange(n_data)
np.random.shuffle(shuffled)
y_shuffled = np.array(y)[shuffled]


# Spectral clustering with single view
scores_single = np.zeros((n_dim,2))
sc= SpectralClustering(3,affinity='precomputed')
for d in range(n_dim):
    y_hat = sc.fit_predict(S[d][shuffled])
    scores_single[d][0] = adjusted_rand_score(y_shuffled,y_hat)
    scores_single[d][1] = normalized_mutual_info_score(y_shuffled,y_hat)

print(pd.DataFrame(scores_single,index=['Spectral Clustering - View '+ str(i) for i in range(n_dim)],columns=['ARI','NMI']))
#%%

m_oi_mlsvd = mc.MC_FR_OI(3,method="mlsvd")
m_oi_hooi = mc.MC_FR_OI(3,method="hooi")
m_mi_direct = mc.MC_FR_MI(3,method="direct",max_iter=500)
m_mi_hooi = mc.MC_FR_MI(3,method="full",max_iter=200)

alg = [m_oi_mlsvd,m_oi_hooi,m_mi_direct]#,m_mi_hooi]

scores = np.zeros((len(alg),2))
for i,a in enumerate(alg) : 
    y_hat = a.fit_predict([S[d][shuffled] for d in range(S.shape[0])])
    scores[i][0] = adjusted_rand_score(y_shuffled,y_hat)
    scores[i][1] = normalized_mutual_info_score(y_shuffled,y_hat)

print(pd.DataFrame(scores,index=alg,columns=['ARI','NMI']))

#%%
# %%