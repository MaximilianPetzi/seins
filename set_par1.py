import numpy as np
N_here=800.

#coarse_grid:
dA_coarse=np.array([36,22,14,8,4])
df_coarse=np.array([.4,1,3,8,18])
deta_coarse=np.array([3,1,.3,.1,.02])

#fine_grid:
dA_fine=np.array([42,28,18,10])
df_fine=np.array([.15,.25,.7,2])
deta_fine=np.array([6,.5,.01])



par_coarse=np.zeros((0,4))
for i in range(len(dA_coarse)):
    for j in range(len(df_coarse)):
        for k in range(len(deta_coarse)):
            par_coarse=np.concatenate((par_coarse,[[dA_coarse[i],df_coarse[j],deta_coarse[k],N_here]]),axis=0)

par_fine=np.zeros((0,4))
for i in range(len(dA_fine)):
    for j in range(len(df_fine)):
        for k in range(len(deta_fine)):
            par_fine=np.concatenate((par_fine,[[dA_fine[i],df_fine[j],deta_fine[k],N_here]]),axis=0)

par_rest=np.zeros((0,4))
for df_rest in (.04,.08):
    for dA_rest in (22,36,60,90,150):
            par_rest=np.concatenate((par_rest,[[dA_rest,df_rest,.5,N_here]]),axis=0)
for df_rest in (.15,.25,.4):
    for dA_rest in (60,90,150):
            par_rest=np.concatenate((par_rest,[[dA_rest,df_rest,.5,N_here]]),axis=0)


par=np.concatenate((par_fine,par_coarse,par_rest),axis=0)
content={"todo":par,"done":np.zeros((0,4))}
print("content=",content)
np.save("paramfile.npy", content)
