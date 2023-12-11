import subgraph_frame_utils as sfu
import scipy.io as io
import numpy as np
import pickle

a = io.loadmat("new_traffic.mat")
adj = a['A'].astype(np.double)
partitions, adjs, children_node_lists, R = sfu.get_partitions(adj)
file = open('trained_filtes/basis_filters_4_20', 'rb')

A,B = pickle.load(file)

file.close()
        
tot_SNR = 0

for j in range(5):
    name = "signals/signal_"+str(j)+'.mat'
    b = io.loadmat(name)
    f = b['f']
    sigma = 1/16 
    b = io.loadmat("signals/16/signal_"+str(j)+'.mat') # folder signals/n contains signals added with noise \sigma = 1/n
    tilde_f = b['f']
    
    
    c, d = sfu.transform(tilde_f,A,B,children_node_lists)

    sfu.hard_thresholding(c,d, 3*sigma)
    f_hat = sfu.inverse(c, d, A, B, children_node_lists)
    
    SNR = 10*np.log10(np.sum(np.square(f))/np.sum(np.square(f-f_hat)))
    tot_SNR += SNR

print("Average SNR",tot_SNR/5)



