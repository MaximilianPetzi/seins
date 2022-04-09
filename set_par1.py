import numpy as np
N_here=800.
#coarse_grid:
#dA=np.array([36,22,14,8,4])
#df=np.array([.4,1,3,8,18])
#deta=np.array([3,1,.3,.1,.02])
#fine_grid:
dA=np.array([42,28,18,10])
df=np.array([.15,.25,.7,2])
deta=np.array([6,.5,.01])

par0=np.zeros((0,4))
for i in range(len(dA)):
    for j in range(len(df)):
        for k in range(len(deta)):
            par0=np.concatenate((par0,[[dA[i],df[j],deta[k],N_here]]),axis=0)

print(np.shape(par0))

content={"todo":par0,"done":np.zeros((0,4))}
print("content=",content)
np.save("paramfile.npy", content)
