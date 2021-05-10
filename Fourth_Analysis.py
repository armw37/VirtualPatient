# Analysis of the s 

import pyximport; pyximport.install()
from frechet import frechet
import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Data Set
with h5py.File('Diabetic_Sim4.h5py', 'r') as data:
    sim_data = {key: value[:,:] for key, value in data.items()}
    

# Create numpy array of data
order_list = [x for x in range(len(sim_data) - 1)]
data_array = np.array([sim_data[str(x)] for x in order_list])

# Order of state_ variables
# t = [:,0]
# V = [:,1]
# H = [:,2]
# L = [:,3]
# I = [:,4]
# M = [:,5]
# F = [:,6]
# R = [:,7]
# E = [:,8]
# P = [:,9]
# A = [:,10]
# S = [:,11]
# D = 1-H-R-I-L

# calculate D
ones_array = np.ones((data_array.shape[0], 
    data_array.shape[1]))
d_mat = ones_array - data_array[:,:,2] - data_array[:,:,7] -\
        data_array[:,:,4] - data_array[:,:,3]
data_array = np.concatenate([data_array, np.atleast_3d(d_mat)], 
        axis=2)

# Calculate number of Deaths
Truth_list = np.any(d_mat >=0.40, axis=1)
d_count = Counter(Truth_list)
d_count['Number of Deaths'] = d_count.pop(True)
d_count['Number of Recoveries'] = d_count.pop(False)
print('The simulation outcomes:')
print(d_count)
print('---------------------------------------------------------')

# max D
max_D = np.amax(d_mat, axis=1)
time_of_max = np.argmax(d_mat, axis=1)

# Max D as a function of params
sample_list= [['gammaV'],['gammaVA'],['gammaVH'],['alphaV'],
        ['aV1'],['aV2'],['bHD'],['aR'],['bHF'],
        ['bIE'],['aI'],['bMV'],['aM'],['bF'],['cF'],['bFH'],
        ['aF'],['bEI'],['bA'],['aA'],['r']]

# to aR
p_dict1 = {param[0]: sim_data['meta_data'][(1+100*i):(101+i*100),i+1] for 
        i, param in enumerate(sample_list[:8])}
max_d_dict1 = {param[0]: max_D[1+100*i:101+i*100] for i, param 
    in enumerate(sample_list[:8])}

# bHF to aI
dict2 = {param[0]: sim_data['meta_data'][(1+100*(i+8)):(101+(i+8)*100),i+10] for 
    i, param in enumerate(sample_list[8:11])}
max_d_dict2 = {param[0]: max_D[1+100*(i+8):101+(i+8)*100] for i, param 
    in enumerate(sample_list[8:11])}

# bMV to aF
dict3 = {param[0]: sim_data['meta_data'][(1+100*(i+11)):(101+(i+11)*100),i+14] for 
        i, param in enumerate(sample_list[11:17])}
max_d_dict3 = {param[0]: max_D[(1+100*(i+11)):(101+(i+11)*100)] for i, param
        in enumerate(sample_list[11:17])}    

# bEI
dict4 = {param: sim_data['meta_data'][(1+100*(i+17)):(101+(i+17)*100),i+21] for 
    i, param in enumerate(sample_list[17])}
max_d_dict4 = {param: max_D[(1+100*(i+17)):(101+(i+17)*100)] for i, param
    in enumerate(sample_list[17])}

# bA
dict5 = {param: sim_data['meta_data'][(1+100*(i+18)):(101+(i+18)*100),i+25] for 
    i, param in enumerate(sample_list[18])}
max_d_dict5 = {param: max_D[(1+100*(i+18)):(101+(i+18)*100)] for i, param
    in enumerate(sample_list[18])}

# Rest
dict6 = {param[0]: sim_data['meta_data'][(1+100*(i+19)):(101+(i+19)*100),i+27] for 
        i, param in enumerate(sample_list[19:])}
max_d_dict6 = {param[0]: max_D[(1+100*(i+19)):(101+(i+19)*100)] for i, param
    in enumerate(sample_list[19:])}

# Make the broader dictionaries
p_dict1 = dict(**p_dict1, **dict2)
max_d_dict1 = dict(**max_d_dict1, **max_d_dict2)

p_dict2 = dict(**dict3, **dict4, **dict5, **dict6)
max_d_dict2 = dict(**max_d_dict3, **max_d_dict4, **max_d_dict5,
        **max_d_dict6)


# Basic Plotting
# first half
fig, axs = plt.subplots(3, 4, sharey=True)
axs = axs.ravel()
for i, item in enumerate(p_dict1.items()):
    param, val = item
    axs[i].scatter(val, max_d_dict1[param], s=4, marker='.')
    axs[i].set_ylabel('Max D')
    axs[i].set_xlabel(param)

plt.tight_layout()

# second half
fig2, axs2 = plt.subplots(3, 4, sharey=True)
axs2 = axs2.ravel()
for i, item in enumerate(p_dict2.items()):
    param, val = item
    axs2[i].scatter(val, max_d_dict2[param], s=4, marker='.')
    axs2[i].set_ylabel('Max D')
    axs2[i].set_xlabel(param)

plt.tight_layout()
plt.show(block=True)

"""
# Plotting the solutions curves for varying parameter of interest values 

interest_list= [['gammaV'],['gammaVA'],['aR'],['gammaHV'],['bHF'],
        ['aI'],['bF'],['bFH'],['aF']]

# create dictionary of interst
tot_dict = {**p_dict1, **p_dict2}
interest_dict = {param[0] :tot_dict[param[0]] for param in interest_list} 

# Get the min, middle and max
argsort_interest = {key: np.argsort(value) for key, value in 
    interest_dict.items()}

points_list = [0, 50, -1]
# index of the parameters of interest
r_index = {'gammaV' : 0, 'gammaVA' : 1, 'aR': 7, 'gammaHV': 8,
        'bHF': 9, 'aI': 11, 'bF': 14, 'bFH': 16 , 'aF': 17}
p_index = {'gammaV' : 1, 'gammaVA' : 2, 'aR': 8, 'gammaHV': 9,
        'bHF': 10, 'aI': 12, 'bF': 16, 'bFH': 18,'aF': 19}

real_index = {key : [(value[i]+1)+(r_index[key]* 100) 
    for i in points_list] for key, value in argsort_interest.items()}
{key : value.append(0) for key, value in real_index.items()}

cols = ['b', 'c', 'm', 'y']
figs = ['fig0', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6',
    'fig7', 'fig8']
axes = ['ax0', 'ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6',
    'ax7', 'ax8']

fig_dict= {}
for i, fig in enumerate(interest_list):
    fig_dict[fig[0]] = plt.subplots(4, 3, sharex=True, figsize=(15,10))

#fig_list = [(fig1, ax1), (fig2, ax2)]

for i, ftup in enumerate(fig_dict.items()):
    key, value = ftup
    fig, axs = value
    for j, sim in enumerate(real_index[key]):
        t = data_array[sim,:,0]
        V = data_array[sim,:,1]*1.5e6
        H = data_array[sim,:,2]
        L = data_array[sim,:,3]
        I = data_array[sim,:,4]
        M = data_array[sim,:,5]
        F = data_array[sim,:,6]
        R = data_array[sim,:,7]
        E = data_array[sim,:,8]
        P = data_array[sim,:,9]
        A = data_array[sim,:,10]
        S = data_array[sim,:,11]
        D = 1-H-R-I-L

        # V
        axs[0, 0].semilogy(t,V, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[0, 0].set_ylabel('V');
        # H
        axs[0, 1].plot(t,H, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[0, 1].set_ylabel('H');
        # L
        axs[3, 1].plot(t,L, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[3, 1].set_ylabel('L')
        # S
        axs[3, 2].plot(t,S, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[3, 2].set_ylabel('S')
        # I
        axs[0, 2].plot(t,I, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[0, 2].set_ylabel('I');
        # M
        axs[1, 0].plot(t,M, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]], 
            3)))
        axs[1, 0].set_ylabel('M');
        # F
        axs[1, 1].semilogy(t,F, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]], 
            3)))
        axs[1, 1].set_ylabel('F');
        # R
        axs[1, 2].plot(t,R, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[1, 2].set_ylabel('R');
        # E
        axs[2, 0].semilogy(t,E, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[2, 0].set_ylabel('E');
        # P
        axs[2, 1].semilogy(t,P, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[2, 1].set_ylabel('P');
        # A
        axs[2, 2].semilogy(t,A, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[2, 2].set_ylabel('A');
        # D
        axs[3, 0].plot(t,D, cols[j], label=key \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[key]],
            3)))
        axs[3, 0].set_ylabel('D')
    
    lg = axs[3, 1].legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.savefig('fig' + str(i) + '.png', bbox_inches='tight',
            bbox_extra_artists=(lg,))


# Calculate the frechet distance in trajectories
dist_list = []
for var in range(1,data_array.shape[2],1):
    state_var_list = []
    for pat in range(1, data_array.shape[0]):
        P = data_array[0,:,var]
        Q = data_array[pat,:,var]
        P = P.copy(order='C')
        Q = Q.copy(order='C')
        f = frechet(P,Q)
        state_var_list.append(f)
    dist_list.append(state_var_list)

disimilar_mat = np.array(dist_list).T

# kmeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(disimilar_mat[:,[0,4,11]])
print('---------------------------------------------------------')
print('The kmeans cluster centers are ')
print(kmeans.cluster_centers_)
print('---------------------------------------------------------')
kmeans_labels = kmeans.labels_

# PCA
pca = PCA(n_components=3).fit(disimilar_mat)
print('The principal components are ')
print('---------------------------------------------------------')
print(pca.components_)

# Graph the kmeans clusters
label_lists = [kmeans_labels[x:x+100] for x in range(0, 
    len(kmeans_labels), 100)]

label_counts = [list(Counter(x).values()) for x in label_lists]

y = [count for label in label_lists for count in list(Counter(label).keys())] 

x = [[i]*len(Counter(label).keys()) for i, label in enumerate(label_lists)]
x = [i for l in x for i in l]

size = [freq for label in label_counts for freq in label]

ticks = [x[0] for x in sample_list]
ticknum = [i for i in range(len(sample_list))]
fig2, axs2 = plt.subplots(1, 1)
axs2.scatter(x, y, s=size)
axs2.set_ylabel('Cluster')
axs2.set_xlabel('Parameter')
axs2.set_yticks([0,1,2])
axs2.set_xticks(ticknum)
axs2.set_xticklabels(ticks, rotation='vertical', visible=True)
plt.tight_layout()

# 3D plot of the diclusters
figk = plt.figure(figsize=(12, 9))
axk = Axes3D(figk, rect=[0, 0, .95, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results
y = kmeans_labels.astype(np.float)
axk.scatter(disimilar_mat[:, 0], disimilar_mat[:, 4], 
        disimilar_mat[:, 11], c=y, edgecolor='k')

axk.set_xlabel('Viral Load difference')
axk.set_ylabel('m Cell difference')
axk.set_zlabel('Dead Cell difference')
axk.set_title('Kmeans Clusters')


plt.show(block=True)
"""
