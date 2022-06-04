###
colored=True      #False, to color according to the errors

#show todo and done in pyhton:
import numpy as np
import os
import matplotlib.cm as cm
params=np.load("paramfile.npy",allow_pickle=True)
todo=params.item().get("todo")
done=params.item().get("done")
print("todo: ",todo)
print("done: ",done)
##### 

#content={"todo":todo,"done":done};np.save("paramfile.npy",content)
####

import matplotlib.pyplot as plt

def sliding_avg(array,alpha):
    ret=[]
    for i in range(np.shape(array)[0]-alpha):
        ret.append(np.average(array[i:i+alpha],axis=0))
    return np.array(ret)

Bfolder="collection/three_grid/all/"

def get_cidx(par,Afolder):
    print("______")
    fname=str(par[0])+"_"+str(par[1])+"_"+str(par[2])+"_"+str(par[3])
    #get error_data
    am=[]
    nrt=10
    for i in range(nrt):
        try:
            trackers = np.load(Bfolder+fname+"/"+str(i+1)+'error.npy')
            if i==1:print(Bfolder+fname+"/"+str(i+1)+'error.npy')
            #print("here:"+str(i+1)+'error.npy')
            t=trackers[0]
            am.append(t[:990])
        except:
            if i==1:print("missing: "+Bfolder+fname+"/"+str(i+1)+'error.npy')

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
    #minerror=np.min(aAm)
    minerror=np.min(sliding_avg(aAm,64))
    enderror=np.average(aAm[-20:])
    #convert into index
    return (enderror-minerror)**1


deta_coarse=np.array([3,1,.3,.1,.02])
deta_fine=np.array([6,.5,.01])
deta_this=deta_coarse   #change this(, line 26) and line 69/71

fig = plt.figure()

for detaidx in range(5):
    subi,subj=int(detaidx/2),detaidx-2*int(detaidx/2)
    ax=plt.subplot(3,2,detaidx+1)
    #ax=axs[subi,subj]
    
    Afolder="sA_20"   
    params=[]
    #only take those parameters of todo that have at least the first simulation completed 
    for i in range(len(todo)):
        parr=todo[i]
        fname=str(parr[0])+"_"+str(parr[1])+"_"+str(parr[2])+"_"+str(parr[3])
        #print("dont append  ", "collection/"+Afolder+"/"+fname+"/1error.npy")
        if os.path.isfile(Bfolder+"/"+fname+"/1error.npy") and parr[2]==deta_this[detaidx]:# and parr[1]<1.5 and parr[0]<80:
            params.append(parr)

    params=np.array(params)


    if not colored:#then, use all parameters in todo, not just the simulated ones
        params=todo

    if not colored:
        cidxs=1
        ppp=ax.scatter(params[:,0],params[:,1])
    if colored:
        cidxs=[get_cidx(para,Afolder) for para in params]
        ppp=ax.scatter(params[:,0],params[:,1],c=cidxs,s=10,cmap=cm.rainbow,vmin = 0, vmax =0.25)
        fig.colorbar(ppp,label="instability")

    ax.set_xlabel('A')
    ax.set_ylabel('f')
    ax.set_title("$\eta$="+str(deta_this[detaidx]))

#fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.6)
plt.show()


