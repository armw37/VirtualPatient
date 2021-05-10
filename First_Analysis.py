# Analysis of the first patient 

import pyximport; pyximport.install()
from frechet import frechet
import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

# Import Data Set
with h5py.File('Diabetic_Sim.h5py', 'r') as data:
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
Truth_list = np.any(d_mat >=0.37, axis=1)
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
bmd_params = sim_data['meta_data'][1:1001,13]
aE_params = sim_data['meta_data'][1001:2001,22]
aP_params = sim_data['meta_data'][2001:3001, 24]

"""
# Plotting
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(bmd_params, max_D[1:1001], s=4, marker='.')
axs[0, 0].set_ylabel('Max D')
axs[0, 0].set_xlabel('bMD')
axs[1, 0].scatter(aE_params, max_D[1001:2001], s=4, marker='.')
axs[1, 0].set_ylabel('Max D')
axs[1, 0].set_xlabel('aE')
axs[0, 1].scatter(aP_params, max_D[2001:3001], s=4, marker='.')
axs[0, 1].set_ylabel('Max D')
axs[0, 1].set_xlabel('aP')
axs[1, 1].hist(max_D, bins=5)
axs[1, 1].set_ylabel('Count Max D')
axs[1, 1].set_xlabel('Max_D')
"""

# Plotting the solutions curves for varying bmd
# loop through varied parameters
"""
# Get the min, middle and max
bmd_sort = np.argsort(bmd_params)
bmds_toplot = [0, bmd_sort[0] + 1, bmd_sort[500] + 1,
        bmd_sort[-1] + 1]

aE_sort = np.argsort(aE_params)
aEs_toplot = [0, aE_sort[0] + 1001, aE_sort[500] + 1001,
        aE_sort[-1] + 1001]

aP_sort = np.argsort(aP_params)
aPs_toplot = [0, aP_sort[0] + 2001, aP_sort[500] + 2001,
        aP_sort[-1] + 2001]

plot_list = [bmds_toplot, aEs_toplot, aPs_toplot]
        
cols = ['b', 'c', 'm', 'y']
p_list = ['bMD', 'aE', 'aP']
p_index = [13, 22, 24]
fig1, ax1 = plt.subplots(4, 3, sharex=True, figsize=(15,10))
fig2, ax2 = plt.subplots(4, 3, sharex=True, figsize=(15,10))
fig3, ax3 = plt.subplots(4, 3, sharex=True, figsize=(15,10))
fig_list = [(fig1, ax1), (fig2, ax2), (fig3, ax3)]

for i, ftup in enumerate(fig_list):
    fig, axs = ftup
    for j, sim in enumerate(plot_list[i]):
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
        axs[0, 0].semilogy(t,V, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[0, 0].set_ylabel('V');
        # H
        axs[0, 1].plot(t,H, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[0, 1].set_ylabel('H');
        # L
        axs[3, 1].plot(t,L, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[3, 1].set_ylabel('L')
        # S
        axs[3, 2].plot(t,S, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[3, 2].set_ylabel('S')
        # I
        axs[0, 2].plot(t,I, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[0, 2].set_ylabel('I');
        # M
        axs[1, 0].plot(t,M, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]], 
            3)))
        axs[1, 0].set_ylabel('M');
        # F
        axs[1, 1].semilogy(t,F, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]], 
            3)))
        axs[1, 1].set_ylabel('F');
        # R
        axs[1, 2].plot(t,R, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[1, 2].set_ylabel('R');
        # E
        axs[2, 0].semilogy(t,E, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[2, 0].set_ylabel('E');
        # P
        axs[2, 1].semilogy(t,P, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[2, 1].set_ylabel('P');
        # A
        axs[2, 2].semilogy(t,A, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[1]],
            3)))
        axs[2, 2].set_ylabel('A');
        # D
        axs[3, 0].plot(t,D, cols[j], label=p_list[i] \
        + '=' + str(round(sim_data['meta_data'][sim,p_index[i]],
            3)))
        axs[3, 0].set_ylabel('D')
    
    lg = axs[3, 1].legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.savefig('fig' + str(i) + '.png', bbox_inches='tight',
            bbox_extra_artists=(lg,))
"""

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
kmeans = KMeans(n_clusters=3, random_state=0).fit(disimilar_mat)
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
fig2, axs2 = plt.subplots(3, 1)
axs2[0].scatter(bmd_params, kmeans_labels[0:1000],
        s=4, marker='.')
axs2[0].set_ylabel('Cluster')
axs2[0].set_xlabel('bMD')
axs2[0].set_yticks([0,1,2])
axs2[1].scatter(aE_params, kmeans_labels[1000:2000],
        s=4, marker='.')
axs2[1].set_ylabel('Cluster')
axs2[1].set_xlabel('aE')
axs2[1].set_yticks([0,1,2])
axs2[2].scatter(aP_params, kmeans_labels[2000:3000], 
        s=4, marker='.')
axs2[2].set_ylabel('Cluster')
axs2[2].set_xlabel('aP')
axs2[2].set_yticks([0,1,2])
plt.tight_layout()
plt.show(block=True)

