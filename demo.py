import subgraph_frame_utils as sfu
import scipy.io as io
import numpy as np
import pickle




if __name__ == "__main__":
    a = io.loadmat("new_traffic.mat")
    adj = a['A'].astype(np.double)
    partitions, adjs, children_node_lists, R = sfu.get_partitions(adj)
    
    
    file = open('trained_filtes/basis_filters_4_20', 'rb')
    
    A,B = pickle.load(file)
    
    file.close()
    
    np.random.seed(2)
     
    SNR = 0
    b = 0
    n = 1
    for i in range(n):
        f = sfu.generate_whole_graph_signal(adj)
        
        c, d = sfu.transform(f,A,B,children_node_lists)
        
        sfu.thresholding(c,d,50)
        
        
        f_hat = sfu.inverse(c,d,A,B,children_node_lists)
        
    
        SNR +=10*np.log10(np.sum(np.square(f))/np.sum(np.square(f-f_hat)))
        b+=np.sum(np.square(f-f_hat)/np.sum(np.square(f)))*100
        
    print("SNR & Relative Error:",SNR/n,b/n)
