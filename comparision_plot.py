import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import matplotlib as mpl


folder = ["1_1_opted_20","4_1_opted_20","Laplacian","n_Laplacian","GraphBior","GraphSS"]

for i in range(len(folder)):
    res = np.zeros(101)
    res[0]=500
    x = np.linspace(0,1,101)
    if i < 4:
        for j in range(5):
            name = "result_comp/"+folder[i]+'/res_'+str(j)+'.npy'
            temp_res = np.load(name)
            # data = {'res':temp_res}
            # io.savemat("result_comp/"+folder[i]+'/res_'+str(j)+'.mat', data)
            for k in range(1,101):
                res[k] += temp_res[k]
        res *= 0.2
        plt.plot(x, res, linestyle='--',label = folder[i])
    else:
        for j in range(5):
            name = "result_comp/"+folder[i]+'/res_'+str(j)+'.mat'
            temp_res = io.loadmat(name)['res']
            print("!!!",temp_res.shape)
            for k in range(1,101):
                res[k] += temp_res[0][k]
        res *= 0.2
        plt.plot(x, res, linestyle='--',label = folder[i])
        

plt.legend()
plt.ylim(0, 50)
plt.show()