import subgraph_frame_utils as sfu
import opt_A
import scipy.io as io
import numpy as np
import pickle



def complement(A):
    u,s,v = np.linalg.svd(A)
    
    m = A.shape[1]
    
    B = u[:,m:]
    
    return B

def get_B_filters(A_list, idx_map, children_node_lists):
    B = []
    A = []
    J = len(children_node_lists)
    
    for  j in range(J):
        temp_A_set = {}
        temp_B_set = {}
        
        for k in children_node_lists[j].keys():
            temp_A = A_list[idx_map[(j,k)]]
            temp_B = complement(temp_A)
            temp_A_set[k] = temp_A.T
            temp_B_set[k] = temp_B.T
        
        A.append(temp_A_set)
        B.append(temp_B_set)
    
    return A,B

if __name__ == "__main__":
    
    a = io.loadmat("new_traffic.mat")
    adj = a['A'].astype(np.double)
    partitions, adjs, children_node_lists, R = sfu.get_partitions(adj)
    
    F = np.zeros((50,1,adj.shape[0]))
    np.random.seed(42)
    for i in range(50):
       
        F[i] = sfu.generate_whole_graph_signal(adj)
        
    R = [1 for x in range(len(R))]
    R[0] = 12
    R[1] = 1
    
    A_list, idx_map = opt_A.get_ortho_matrix(F, children_node_lists, R)
    A, B = get_B_filters(A_list, idx_map, children_node_lists)
    filters = [A,B]
    
    file = open('trained_filters/basis_filters_'+str(R[0]), 'wb')
    pickle.dump(filters, file)
    file.close()
    