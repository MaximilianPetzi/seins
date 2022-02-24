import numpy as np

params=np.load("paramfile.npy",allow_pickle=True)
params=params.item().get("todo")
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#0=A, 1=f, 2=eta
print(np.shape(params[0]),np.shape(params[1]),np.shape(params[2]))

#ax.scatter(params[:,0]+2,params[:,1],params[:,2],color="red")
ax.scatter(params[:,0],params[:,1],params[:,2],color="black")
#ax.scatter(params[:,0]-2,params[:,1],params[:,2],color="green")
#ax.scatter(params[:,0]-4,params[:,1],params[:,2],color="blue")

ax.set_xlabel('A')
ax.set_ylabel('f')
ax.set_zlabel('$\eta$')

plt.show()


