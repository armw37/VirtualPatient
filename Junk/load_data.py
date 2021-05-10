# Load Covid Immune model data

import h5py
import tarfile
from os import listdir, getcwd, mkdir
from os.path import isfile, join 
import numpy as np

# extract
tar_name = 'Diabetic_Sim.tar.xz'
data_file = 'Diabetic_Sim.h5py'

onlyfiles = [f for f in listdir(getcwd()) if 
        isfile(join(getcwd(),f))]
if data_file not in onlyfiles:
    tf = tarfile.open(tar_name)
    tf.extract(data_file)
    tf.close()

with h5py.File(data_file, 'r') as data:
    sim_data = {key: value[:,:] for key, value in data.items()}

# save as csv
newdir = 'simulation_data'
if newdir not in listdir(getcwd()):
    mkdir(getcwd() + '/simulation_data')
    for key, value in sim_data.items():
        filepath = getcwd() + '/simulation_data/patient_'
        full = filepath + key + '.csv'
        np.savetxt(full, value, delimiter=',')
