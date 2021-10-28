import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
#matplotlib.use("TkAgg")
import os
os.system("rm bilder/plit_temp/*")
nrt=25
am = []
for i in range(nrt):
    print(i+1)
    try:
        t = np.load("error_h/"+str(i+1)+'error.npy')
        am.append(t)
    except:
        print("--"+str(i+1)+"error.npy is missing--")
am=np.array(am)
nrtries=len(am)

minl=10000000
for i in range(len(am)):
    minl=np.min([minl,len(am[i])])
Am=np.zeros((len(am),minl))
for i in range(len(am)):
    amm=am[i]
    Am[i,:]=amm[:minl]
print("data-shape before avg:",np.shape(Am))

#sliding avg:
siz=15
slav=np.zeros(np.shape(Am[:,siz-1:]))
for i in range(len(slav[0])):
    slav[:,i]=slav[:,i]+np.sum(Am[:,i:i+siz],axis=1)
slav=slav/siz
#error and error of error   last Navg_last trials:
Navg_last=50
lasterrs=np.average(Am[:,-Navg_last:],axis=1)
lasterr=np.average(lasterrs)
lasterrerr=np.std(lasterrs)/np.shape(lasterrs)[0]**.5
print("lasterr=",lasterr, "+/-",lasterrerr)
print("lasterrs",lasterrs)

yerr=np.std(Am,axis=0)/nrtries**.5
#sliding_av=np.average(slav,axis=0)
aAm=np.average(Am,axis=0)


#plt.plot(sliding_av,color=(1,0,0,1),label="sliding avg")
plt.plot(np.array(Am).T)
plt.figure()
plt.plot(aAm,color="black",linewidth=.4)
plt.errorbar(x=range(len(aAm)),y=aAm, yerr=yerr,color=(1,0,0,.2),elinewidth=.5)
plt.ylabel("error avg over "+str(nrtries)+" identical tries")
plt.legend()
plt.savefig("bilder/plit_temp/this")
plt.show()

