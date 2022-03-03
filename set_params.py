import numpy as np

dA=np.array([20,18,16,14,12,10])
df=np.array([1.5,2,3,4.5,7,9])

deta=np.array([1,.8,.6,.4,.2,.1])
fn=np.array([])
etan=np.array([])
An=np.array([])
linelen=6
#triangle plane: A 10 == f 1.5 == eta 0.1 
for i in range(linelen):
    if i==0:An=np.append(An,dA)
    An=np.append(An,dA[:-i])
    fn=np.append(fn,df[i:])
    etan=np.append(etan,np.ones(linelen-i)*deta[i])

An=dA
fn=df 
etan=deta

Nn=np.ones(len(An))*800.
par01=np.array([An+2,fn,etan,Nn]).T
par0=np.array([An,fn,etan,Nn]).T
par1=np.array([An-2,fn,etan,Nn]).T
par2=np.array([An-4,fn,etan,Nn]).T


#par=np.concatenate([par01,par0,par1,par2],axis=0)

content={"todo":par0,"done":np.zeros((0,4))}
print("content=",content)
np.save("paramfile.npy", content)
