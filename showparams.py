###
justpars=True       #False, to color according to the errors

#show todo and done in pyhton:
import numpy as np
import os
import matplotlib.cm as cm
params=np.load("paramfile.npy",allow_pickle=True)
todo=params.item().get("todo")
done=params.item().get("done")
print("todo: ",todo)
print("done: ",done)

import matplotlib.pyplot as plt

def sliding_avg(array,alpha):
    ret=[]
    for i in range(np.shape(array)[0]-alpha):
        ret.append(np.average(array[i:i+alpha],axis=0))
    return np.array(ret)

def get_cidx(par,Afolder):
    fname=str(par[0])+"_"+str(par[1])+"_"+str(par[2])+"_"+str(par[3])
    #get error_data
    am=[]
    nrt=10
    for i in range(nrt):
        #print(i+1)
        try:
            trackers = np.load("collection/"+Afolder+"/"+fname+"/"+str(i+1)+'error.npy')
            #print("here:"+str(i+1)+'error.npy')
            t=trackers[0]
            am.append(t[:990])
        except:
            pass
            print("missing: collection/"+Afolder+"/"+fname+"/"+str(i+1)+'error.npy')

            #print("missing: varA_errors/"+strlist[tracker]+str(i+1)+'error.npy')
    am=np.array(am)

    minl=10000000
    for i in range(len(am)):
        minl=np.min([minl,len(am[i])])
    Am=np.zeros((len(am),minl))
    for i in range(len(am)):
        amm=am[i]
        Am[i,:]=amm[:minl]
    #print("data-shape before avg:",np.shape(Am))
    aAm=np.average(Am[:18],axis=0)
    minerror=np.min(sliding_avg(aAm,64))
    #convert into index
    return minerror

Afolder="sA_20"   
params=[]
#only take those parameters of todo, that have at least the first simulation completed 
for i in range(len(todo)):
    parr=todo[i]
    fname=str(parr[0])+"_"+str(parr[1])+"_"+str(parr[2])+"_"+str(parr[3])
    #print("dont append  ", "collection/"+Afolder+"/"+fname+"/1error.npy")
    if os.path.isfile("collection/"+Afolder+"/"+fname+"/1error.npy"):
        params.append(parr)

params=np.array(params)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#0=A, 1=f, 2=eta
if justpars:
    params=todo
print(np.shape(params[0]),np.shape(params[1]),np.shape(params[2]))


if justpars:
    cidxs=1
    ppp=ax.scatter(params[:,0],params[:,1],params[:,2])
if not justpars:
    cidxs=[get_cidx(para,Afolder) for para in params]
    ppp=ax.scatter(params[:,0],params[:,1],params[:,2],c=cidxs,cmap=cm.rainbow)
    fig.colorbar(ppp)
#ax.scatter(params[:,0]-2,params[:,1],params[:,2],color="green")
#ax.scatter(params[:,0]-4,params[:,1],params[:,2],color="blue")

ax.set_xlabel('A')
ax.set_ylabel('f')
ax.set_zlabel('$\eta$')

plt.show()


