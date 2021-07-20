import numpy as np
import h5py
import sys

if(len(sys.argv)<2):
    print("wrong usage: bin2h5.py [binary_filename] [h5_filename] ")
    
input_and_output_dim=138
bin_file_name=sys.argv[1]
data = np.fromfile(bin_file_name, dtype='float32')
data = np.reshape(data, (len(data)//input_and_output_dim, input_and_output_dim))
h5f = h5py.File(sys.argv[2], 'w');
h5f.create_dataset('data', data=data)
h5f.close()