
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt

nrt=10
am=[]
for i in range(nrt):
    t = np.load("collection/sA_20/18.0_3.0_0.8_800.0/"+str(i+1)+'error.npy')
    #print("here:"+str(i+1)+'error.npy')
    am.append(t[0])
    #plt.plot(t[0],linewidth=.4)


am=np.array(am)
amavg=np.average(am,axis=0)
plt.plot(amavg,color="black",linewidth=2)
yerrs=np.std(am,axis=0)/(nrt)**.5
plt.errorbar(x=np.arange(len(amavg)),y=amavg, yerr=yerrs,color="black",linewidth=.5)
plt.show()
    