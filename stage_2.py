import subgraph_frame_utils as sfu
import opt_B
import scipy.io as io
import numpy as np
import pickle
import time


def l1_norm(x):
    return np.sum(np.abs(x))

def l2_norm(x):
    return np.sum(np.square(x))

def collect_norm(F, A, B, children_node_lists, N):
    d_norm_list = {}
    
    for i in range(F.shape[0]):
        c,d = sfu.transform(F[i], A, B, children_node_lists)
        for j in range(len(d)):
            for k in d[j].keys():
                if (j,k) not in d_norm_list.keys():
                    d_norm_list[(j,k)] = 0.0
                d_norm_list[(j,k)] += l2_norm(d[j][k])
    
    temp_list = []
    for k in d_norm_list.keys():
        temp_list.append((d_norm_list[k],k))
    
    res = sorted(temp_list, key = lambda x: x[0], reverse = True)
    
    return set([x[1] for x in res[0:N]])
    
def select_filters(A,B,F,partitions,adjs,children_node_lists, R, tree_node_id, frame = False):  # frame set to True to generate frames
    c = [[] for x in range(F.shape[0])]
    
    temp_c = [{} for x in range(F.shape[0])]
    for j in range(F.shape[0]):
        for i in range(F.shape[2]):
            temp_c[j][i] = np.zeros((1,1))
            temp_c[j][i][0,0] = F[j][0,i]
        c[j].append(temp_c[j])
    

    J = len(children_node_lists)
    
    for  j in range(J):
        temp_c = [{} for x in range(F.shape[0])]
        temp_d = [{} for x in range(F.shape[0])]
        temp_d_2 = [{} for x in range(F.shape[0])]
        for k in children_node_lists[j].keys():
            
            temp_A = A[j][k] 
            temp_B = B[j][k]
            temp_B_f = sfu.spectral_frame(B[j][k])
            sum_1 = 0.0
            
            a = 0
            
            C_1 = []
            C_2 = []
            for i in range(F.shape[0]):
                tot = len(children_node_lists[j][k])
                s = c[i][j][children_node_lists[j][k][0]].shape[1]
                temp_C = np.zeros((tot,s))
                for l in range(tot):
                    temp_C[l,:] = c[i][j][children_node_lists[j][k][l]]
                    
                temp_c[i][k] = np.dot(temp_A,temp_C).reshape(1,-1)
                temp_d[i][k] = np.dot(temp_B,temp_C).reshape(1,-1)
                temp_d_2[i][k] = np.dot(temp_B_f,temp_C).reshape(1,-1)
                
                if (j,k) in tree_node_id:
                    C_1.append(np.dot(temp_B,temp_C).T)
                    C_2.append(np.dot(temp_B_f,temp_C).T)
                
                sum_1 += l1_norm(temp_d[i][k])
                
                
                a += l2_norm(temp_d[i][k])
                
            
            if (j,k) in tree_node_id:
                if not frame:
                    p1, res1 = opt_B.get_ortho_matrix(C_1)
                    B[j][k] = p1.T@temp_B
                else:
                    p1, res1 = opt_B.get_ortho_matrix(C_2)
                    B[j][k] = p1.T@temp_B_f
            else:
                if not frame:
                    B[j][k] = temp_B
                else:
                    B[j][k] = temp_B_f
        
        for i in range(F.shape[0]):
            c[i].append(temp_c[i])
        
       
    return A,B

if __name__ == "__main__":
    
    a = io.loadmat("new_traffic.mat")
    adj = a['A'].astype(np.double)
    partitions, adjs, children_node_lists, R = sfu.get_partitions(adj)
    
    
    F = np.zeros((50,1,adj.shape[0]))
    np.random.seed(42)
    for i in range(50):
        
        F[i] = sfu.generate_whole_graph_signal(adj)
       
    file = open('trained_filters/basis_filters_1', 'rb')
    
    A,B = pickle.load(file)
    
    file.close()
    
    
            
    tree_node_id = collect_norm(F, A, B, children_node_lists, 100) # the final paramter indicates the top-N subspaces to be selected
    s = time.time()
    A, B = select_filters(A, B, F, partitions, adjs, children_node_lists, R, tree_node_id)
    t = time.time()
    print("TIME:",t-s)
    filters = [A,B]
    
    file = open('trained_filters/basis_filters_1_100', 'wb')
    pickle.dump(filters, file)
    file.close()
    
    