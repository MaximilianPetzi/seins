import numpy as np

An=np.array([20.,15,10,20])#,12,10])
fn=np.array([9,9,9,1.5])#,7,9])
#fn=np.array([1.5,2,3,4.5])#,7,9])
etan=np.ones(len(An))
Nn=np.ones(len(An))*800.
par01=np.array([An+2,fn,etan,Nn]).T
par0=np.array([An,fn,etan,Nn]).T
par1=np.array([An-2,fn,etan,Nn]).T
par2=np.array([An-4,fn,etan,Nn]).T


#par=np.concatenate([par01,par0,par1,par2],axis=0)

content={"todo":par0,"done":np.zeros((0,4))}
np.save("paramfile.npy", content)
