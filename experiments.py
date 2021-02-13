#%%
# %load_ext autoreload
# %autoreload 2

import multiview_clustering as mc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import pandas as pd
from utils import SpectralClustering
#%%[markdown]
# ## Synthetic data creation

#%%
N_DATA = 350
N_DIM = 4

def create_synth_data(data_type='bool',noise_factor = 0):
    n_data = N_DATA
    n_dim = N_DIM

    S = np.zeros((n_dim,n_data,n_data))# Similarity tensor
    cluster_members = [50,100,200]

    # Contruct labels
    y = [] 
    for i,c in enumerate(cluster_members):
        y.extend(np.ones(c)*i) 

    intra_c_prob = np.array([[0.8,0.1,0.7],[0.05,0.4,0.06],[0.6,0.8,0.67],[0.6,0.4,0.1]])
    # noise_d = np.array([0.25,0.15,0.45,0.22]) + noise_factor
    noise_d = np.array([0,0,0,0]) + noise_factor 
    intra_cluster_proba = 0.1

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
            S[d] = S[d] + np.random.normal(loc=0,scale=noise_d[d],size=S[d].size).reshape(S[d].shape)
        else :
            raise ValueError

        # Make symmetric
        S[d] = (np.tril( S[d]) + np.triu(S[d].T, 1))

        S[d] = np.where(S[d]<0,0,S[d]) # remove negative

        if data_type == "bool" :
            S[d] = np.where(S[d]>1,1,S[d]) # Remove val > 1
        elif data_type == "linear":
            S[d] = np.where(S[d]>1,0,S[d]) # Remove val > 1

        # Set diag to single value
        di,dj = np.diag_indices_from(S[d])
        S[d][di,dj] = 1
    
    return S,y


# %%

data_type = 'bool' # or linear

S,y= create_synth_data(data_type=data_type,noise_factor=0)

fig,ax = plt.subplots(1,N_DIM,figsize=(13,4))
for d in range(N_DIM):
    ax[d].imshow(S[d])

fig.suptitle("Similarity Matrices")
plt.show()

#%%

shuffled = np.arange(N_DATA)
np.random.shuffle(shuffled)
y_shuffled = np.array(y)[shuffled]

# Spectral clustering with single view
scores_single = np.zeros((N_DIM,2))
sc= SpectralClustering(3)
for d in range(N_DIM):
    y_hat = sc.fit_predict(S[d][shuffled][:,shuffled])
    scores_single[d][0] = adjusted_rand_score(y_shuffled,y_hat)
    scores_single[d][1] = normalized_mutual_info_score(y_shuffled,y_hat)

print(pd.DataFrame(scores_single,index=['Spectral Clustering - View '+ str(i) for i in range(N_DIM)],columns=['ARI','NMI']))
#%%

m_oi_mlsvd = mc.MC_FR_OI(3,method="mlsvd")
m_oi_hooi = mc.MC_FR_OI(3,method="hooi")
m_mi_direct = mc.MC_FR_MI(3,method="direct",max_iter=100)
m_mi_hooi = mc.MC_FR_MI(3,method="full",max_iter=100)

alg = [m_oi_mlsvd,m_oi_hooi,m_mi_direct,m_mi_hooi]

scores = np.zeros((len(alg),2))
for i,a in enumerate(alg) : 
    y_hat = a.fit_predict([S[d][shuffled][:,shuffled]for d in range(S.shape[0])])
    scores[i][0] = adjusted_rand_score(y_shuffled,y_hat)
    scores[i][1] = normalized_mutual_info_score(y_shuffled,y_hat)

print(pd.DataFrame(scores,index=alg,columns=['ARI','NMI']))

# %%
########################## With some varying noise ############################

alg = [m_oi_mlsvd,m_oi_hooi]

score_global = []
# range_n_linear = np.linspace(-0.15,1,20)
# range_n_bool = np.linspace(-0.15,0.55,20)
range_n_linear = np.linspace(0,1,10)
range_n_bool = np.linspace(0,1,10)
ranges = [range_n_linear,range_n_bool]
data_type_l = ['linear','bool']
n_runs = 5

ii = 0
nn = n_runs * (len(range_n_linear) + len(range_n_bool) )

for ri,r in  enumerate(ranges):
    range_n = r
    score_runs = np.zeros((len(range_n),4+1+len(alg),2))
    for n in range(n_runs):
        score_subglobal = []
        for noise_r in range_n:
            print(f"{ii+1} / {nn}")
            S,y = create_synth_data(data_type_l[ri],noise_factor=noise_r)

            shuffled = np.arange(S.shape[1])
            np.random.shuffle(shuffled)
            y_shuffled = np.array(y)[shuffled]

            scores = np.zeros((S.shape[0]+1+len(alg),2))

            sc= SpectralClustering(3)

            ## Single View Spectral 
            for d in range(S.shape[0]):
                y_hat = sc.fit_predict(S[d][shuffled][:,shuffled])
                scores[d][0] = adjusted_rand_score(y_shuffled,y_hat)
                scores[d][1] = normalized_mutual_info_score(y_shuffled,y_hat)
            
            ## Kernel Fusion
            d+=1
            Sf= S.mean(axis=0)
            y_hat = sc.fit_predict(Sf[shuffled][:,shuffled])
            scores[d][0] = adjusted_rand_score(y_shuffled,y_hat)
            scores[d][1] = normalized_mutual_info_score(y_shuffled,y_hat)
            

            for i in range(len(alg)) :
                y_hat = alg[i].fit_predict([S[c][shuffled][:,shuffled]for c in range(S.shape[0])])
                scores[d+1+i][0] = adjusted_rand_score(y_shuffled,y_hat)
                scores[d+1+i][1] = normalized_mutual_info_score(y_shuffled,y_hat)
            
            ii += 1
            
            score_subglobal.append(scores)
        score_runs = score_runs + np.array(score_subglobal) 
    score_runs = score_runs /n_runs

    score_global.append(score_runs)

idx=  ['Spectral Clustering - View '+ str(i) for i in range(S.shape[0])]
idx.extend(['Kernel Fusion'])
idx.extend(alg)
score_names = ['ARI','NMI']

# %%
scores_np = np.array(score_global)


fig,ax = plt.subplots(2,len(score_names),figsize=(15,10))
for ri in range(len(data_type_l)):
    if data_type_l[ri] == 'bool':
        titl = 'using Adjancy Matrices'
    elif data_type_l[ri] == 'linear':
        titl = 'using Similarity Matrices'
    else : 
        titl = ''

    for i,s in enumerate(score_names):
        ax[ri,i].set_title(f"{s} {titl}")
        ax[ri,i].set_xlabel("added noise")
        ax[ri,i].set_ylabel(s)

        for j in range(len(idx)):
            ax[ri,i].plot(ranges[ri],scores_np[ri,:,j,i],label=idx[j])


plt.legend()
plt.tight_layout()
# plt.savefig("fig-withbasenoise-30pts-10runs.png")
plt.show()
#%%

