import subgraph_frame_utils as sfu
import scipy.io as io
import numpy as np
import pickle
import copy


a = io.loadmat("new_traffic.mat")
adj = a['A'].astype(np.double)
b = io.loadmat("signals/signal_4.mat")
f = b['f'];

partitions, adjs, children_node_lists, R = sfu.get_partitions(adj)



file = open('trained_filters/basis_filters_4_20', 'rb')

A,B = pickle.load(file)

file.close()

z = np.linspace(0,1,101)
c, d = sfu.transform(f,A,B,children_node_lists)
res = np.zeros(101);

for k in range(1,101):
    K = int(np.ceil(z[k]*f.shape[1]))-1
    temp_c = copy.deepcopy(c)
    temp_d = copy.deepcopy(d)
    sfu.thresholding(temp_c,temp_d,K)
    f_hat = sfu.inverse(temp_c,temp_d,A,B,children_node_lists)
    RE = np.sum(np.square(f-f_hat)/np.sum(np.square(f)))*100
    res[k] = RE

data = {'res':res}
io.savemat('nla_result/4_20/res_'+str(4)+'.mat', data)