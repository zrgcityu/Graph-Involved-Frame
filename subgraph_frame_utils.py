from sknetwork.hierarchy import LouvainHierarchy, cut_balanced
import numpy as np
import scipy.io as io
import networkx as nx
import clustering as cl


def my_save(f,idx,adj,name):
    
    data = {'f':f,'idx':np.array(idx)+1, 'A': adj}
    
    io.savemat(name+'.mat',data)

def Laplacian(adj):
    
    D = np.diag(np.array(np.sum(adj,axis = 1)).flatten())
    L = D - adj
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    return eigenvalues, eigenvectors

def n_Laplacian(adj):
    D = np.diag(np.power(np.array(np.sum(adj,axis = 1)).flatten(),-0.5))
    L = np.eye(adj.shape[0]) - D@adj@D
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors

def get_partitions(adj):
    
    
    print("******Generating partition tree******")
    n = adj.shape[0]
    
    cluster_size_bound = 15
    
    method = LouvainHierarchy(random_state = 1)
    partitions= []
    adjs = [adj]
    children_node_lists = []
    partitions.append([i for i in range(n)])
    prev_cluster_set = {}
    
    R =[]
    
    for i in range(n):
        temp_set = set()
        temp_set.add(i)
        prev_cluster_set[i] = temp_set
    
    prev_adj = adj
    
    while(prev_adj.shape[0]>cluster_size_bound+1 and np.abs(np.sum(prev_adj))>1e-6):
    
        dendrogram = method.fit_transform(prev_adj)
        cluster_id = cut_balanced(dendrogram, cluster_size_bound)
        cluster_id = cl.postprocess_merge(prev_adj, cluster_id, cluster_size_bound)
        
        cluster_num = max(cluster_id) + 1
        print("Number of clusters at current level:",cluster_num)
        
        
        
        temp_cluster_set = {}                    #\mathcal{S}_{j,k}
        temp_cluster_list = {}                   #\mathcal{C}_{j,k}
        
        for j in range(len(cluster_id)):
            temp_id = cluster_id[j]
            if temp_id not in temp_cluster_set.keys():
                temp_cluster_set[temp_id] = prev_cluster_set[j]
                temp_cluster_list[temp_id] = [j]
            else:
                temp_cluster_set[temp_id] = temp_cluster_set[temp_id].union(prev_cluster_set[j])
                temp_cluster_list[temp_id].append(j)
        
        
        min_size = 10000
        
        for k in temp_cluster_list.keys():
            min_size = min(min_size, len(temp_cluster_list[k]))
        
        R.append(min_size)
        
        children_node_lists.append(temp_cluster_list)
        
        
        prev_cluster_set = temp_cluster_set
        temp_adj = np.zeros((cluster_num,cluster_num))
        
        for j in range(cluster_num):
            for k in range(j+1,cluster_num):
                
                edge_weight = 0.0
                
                for p in temp_cluster_list[j]:
                    for q in temp_cluster_list[k]:
                        edge_weight += prev_adj[p][q]
                
                temp_adj[j][k] += edge_weight
                temp_adj[k][j] += edge_weight
        
        adjs.append(temp_adj)
        prev_adj = temp_adj
        
        temp_partition = [0 for j in range(n)]
        for j in range(cluster_num):
            temp_list = list(temp_cluster_set[j])
            for p in temp_list:
                temp_partition[p] = j
        
        partitions.append(temp_partition)
    
    if prev_adj.shape[0]>1:
        temp_cluster_list = {}
        temp_cluster_list[0] = [x for x in range(prev_adj.shape[0])];
        children_node_lists.append(temp_cluster_list)
        
        min_size = 10000
        
        for k in temp_cluster_list.keys():
            min_size = min(min_size, len(temp_cluster_list[k]))
        
        R.append(min_size)
    
    print("*******Generation completed******")
    return partitions, adjs, children_node_lists, R


def parent_nodes(rooted_T_edges, num_nodes):
    p = [0 for x in range(num_nodes)]
    for e in rooted_T_edges:
        p[e[1]] = e[0]
    
    return p

def depth(u, p):
    if u == 0:
        return 0
    
    return depth(p[u],p) + 1

def nu(x):
    
    x = np.power(x,4)*(35-84*x+70*np.power(x,2)-20*np.power(x,3))
    
    return x

def alpha(x):
    
    temp = np.zeros(x.shape)
    
    temp[np.abs(x)<0.25] = 1
    
    
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)] = \
        np.cos(np.pi/2*nu(4*np.abs(x)-1))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)]
        
    return temp

def beta_1(x):
    
    temp = np.zeros(x.shape)
    
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)] = \
        np.sin(np.pi/2*nu(4*np.abs(x)-1))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)]
        
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)] = \
        np.power(np.cos(np.pi/2*nu(2*np.abs(x)-1)),2)\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)]
    
    return temp

def beta_2(x):
    temp = np.zeros(x.shape)
        
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)] = \
        (np.cos(np.pi/2*nu(2*np.abs(x)-1))*np.sin(np.pi/2*nu(2*np.abs(x)-1)))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)]
    
    return temp

def spectral_frame(B, J = 2):
    
    tot = B.shape[0]
    all_id = [x for x in range(B.shape[1])]
    
    temp_B = np.zeros((3*J*B.shape[1],B.shape[1]))
    
    x = np.linspace(0,1,tot)
    cnt = 0;
    for j in range(1,J+1):
        temp_x = x/np.power(2,j-1)
        
        temp_vec_1 = alpha(temp_x)
        temp_vec_2 = beta_1(temp_x)
        temp_vec_3 = beta_2(temp_x)
        
        
        for k in range(B.shape[1]):
            u = B[:,k]
            if j == 1:
                a = np.expand_dims(temp_vec_1*u, axis=0)
                temp_B[np.ix_([cnt],all_id)] = np.dot(a,B)
                cnt += 1
            b_1 = np.expand_dims(temp_vec_2*u, axis=0)
            temp_B[np.ix_([cnt],all_id)] = np.dot(b_1,B)
            cnt += 1
            b_2 = np.expand_dims(temp_vec_3*u, axis=0)
            temp_B[np.ix_([cnt],all_id)] = np.dot(b_2,B)
            cnt += 1
            
    return temp_B
    
def get_low_highpass(sub_adj, r, spectral = False):
    
    eigenvalues, eigenvectors = Laplacian(sub_adj)
        
    
    all_id = [x for x in range(sub_adj.shape[0])]
    selected_id = [x for x in range(r)]
    remaining_id = [x for x in range(r,sub_adj.shape[0])]
    
    A = np.transpose(eigenvectors[np.ix_(all_id,selected_id)])
    B = np.transpose(eigenvectors[np.ix_(all_id,remaining_id)])
    
    if spectral is True and B.shape[0]>= 4:
        temp_B = spectral_frame(B)
        B = temp_B
    
    return A, B

def get_filters(partitions,adjs,children_node_lists, R, spectral = False):
    A = []
    B = []
    J = len(children_node_lists)
    
    for  j in range(J):
        temp_A_set = {}
        temp_B_set = {}
        
        for k in children_node_lists[j].keys():
            children_id = children_node_lists[j][k]
            sub_adj = adjs[j][np.ix_(children_id,children_id)]
            
            temp_A, temp_B = get_low_highpass(sub_adj,R[j],spectral)
            
            temp_A_set[k] = temp_A
            temp_B_set[k] = temp_B
        
        A.append(temp_A_set)
        B.append(temp_B_set)
        
        
    return A,B

def transform(f,A,B,children_node_lists):
    c = []
    d = []
    temp_c = {}
    for i in range(f.shape[1]):
        temp_c[i] = np.zeros((1,1))
        temp_c[i][0,0] = f[0,i]
    c.append(temp_c)
    
    J = len(A)
    
    for j in range(J):
        temp_c = {}
        temp_d = {}
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            R = c[j][children_node_lists[j][k][0]].shape[1]
            
            temp_C = np.zeros((tot,R))
            
            for l in range(tot):
                temp_C[l,:] = c[j][children_node_lists[j][k][l]]
                
            temp_c[k] = np.dot(A[j][k],temp_C).reshape(1,-1)
            temp_d[k] = np.dot(B[j][k],temp_C).reshape(1,-1)
         
        c.append(temp_c)
        d.append(temp_d)
    
    return c[-1], d
            
def inverse(c,d,A,B,children_node_lists):
    
    J = len(d)
    
    for j in range(J-1,-1,-1):
        new_c = {}
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            r = A[j][k].shape[0]
            m = B[j][k].shape[0]
            # print("!!!",r,m)
            
            temp_c = np.concatenate(np.split(c[k],r,axis = 1), axis = 0)
            temp_d = np.concatenate(np.split(d[j][k],m,axis = 1), axis = 0)

            temp_c = np.dot(np.transpose(A[j][k]),temp_c)
            temp_d = np.dot(np.transpose(B[j][k]),temp_d)
            
            temp_C = temp_c + temp_d
            all_id = [x for x in range(temp_C.shape[1])]
            for l in range(tot):
                new_c[children_node_lists[j][k][l]] = temp_C[np.ix_([l],all_id)]
        
        c = new_c
    
    f = np.zeros((1,len(list(c.keys()))))
    for i in c.keys():
        f[0][i] = c[i][0][0]
    
    return f


def generate_whole_graph_signal(adj,name="",poly_deg = 9):
    G = nx.from_numpy_array(adj)

    T = nx.minimum_spanning_tree(G)
    rooted_T = nx.dfs_tree(T,source = 0)
    rooted_T_edges = list(rooted_T.edges())
    p = parent_nodes(rooted_T_edges, adj.shape[0])
    
    dep = []
    for i in range(adj.shape[0]):
        dep.append(depth(i,p))
    
    max_dep = max(dep)
    c = np.random.rand(poly_deg+1)
    x = np.linspace(-1,1,num = max_dep)
    vals = np.polynomial.chebyshev.chebval(x,c)
    
    f = np.zeros((1,adj.shape[0]))
    for i in range(adj.shape[0]):
        f[0,i]  = vals[dep[i]-1]
    # idx = [x for x in range(adj.shape[0])]
    # my_save(f,idx,adj,name)
    return f


def generate_whole_graph_sine_signal(adj):
    G = nx.from_numpy_array(adj)
    T = nx.minimum_spanning_tree(G)
    rooted_T = nx.dfs_tree(T,source = 0)
    rooted_T_edges = list(rooted_T.edges())
    p = parent_nodes(rooted_T_edges, adj.shape[0])
    
    dep = []
    for i in range(adj.shape[0]):
        dep.append(depth(i,p))
    
    max_dep = max(dep)
    print("@@@",max_dep)
    x = np.linspace(0,0.2,num = max_dep)
    vals = np.sin(np.power(0.01+2*x,-1))
    
    temp_f = np.zeros((1,adj.shape[0]))
    for i in range(adj.shape[0]):
        temp_f[0,i]  = vals[dep[i]-1]
    return temp_f



def thresholding(c, d, N):
    f = []
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            f.append((c[k][0][l],(1,k,l)))
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                f.append((d[j][k][0][l],(2,j,k,l)))
    
    
    
    temp_f = sorted(f,key = lambda x: np.abs(x[0]),reverse=True)
    
    
    # threshold = np.abs(temp_f[N][0])
    
    idx_set = set([x[1] for x in temp_f[0:N]])
        
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            if (1,k,l) not in idx_set:
                c[k][0][l] = 0
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                if (2,j,k,l) not in idx_set:
                    d[j][k][0][l] = 0

def hard_thresholding(c,d,sigma):
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            if abs(c[k][0][l])-sigma > 0:
                continue
            c[k][0][l] = 0
            
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                if abs(d[j][k][0][l])-sigma > 0:
                    continue
                d[j][k][0][l] = 0
                
    
    

def save_coeff(c, d, name = '1-1', is_mat = True):
    f = []
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            f.append((c[k][0][l]))
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                f.append((d[j][k][0][l]))
    
    if not is_mat:
        with open('coeff/'+name+'.npy','wb') as file:
            np.save(file,np.array(f))    
    else:
        data = {'f':np.array(f)}
        io.savemat(name+'.mat',data)

    