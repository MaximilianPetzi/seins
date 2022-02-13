import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
import matplotlib.cm as cm
#matplotlib.use("TkAgg")
import os
num_goals=8

def sliding_avg(array,alpha):
    ret=[]
    for i in range(np.shape(array)[0]-alpha):
        ret.append(np.average(array[i:i+alpha],axis=0))
    return np.array(ret)

[error_h,g_h,etaf_h]=np.load("error_h_goals/1error.npy")
for g in range(8):
    graph=error_h[g::num_goals]
    #plt.plot(graph,linewidth=.4)
    plt.plot(sliding_avg(graph,8),linewidth=.4)



plt.show()

