import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
import matplotlib.cm as cm
#matplotlib.use("TkAgg")
import os
import sys
#import pandas as pd
os.system("rm bilder/plit_temp/*")
nrt=25
aAms=[]
yerrs=[]
strlist=["error_h_A14/","error_h_A10/","error_h_A14-A10after40/"]
for tracker in range(3):
    am = []
    for i in range(nrt):
        #print(i+1)
        try:
            trackers = np.load("saved_errors/"+strlist[tracker]+str(i+1)+'error.npy')#trackers = np.load("error_h/"+str(i+1)+'error.npy')
            t=trackers[0]
            am.append(t[:990])
        except:
            print("saved_errors/"+strlist[tracker]+str(i+1)+'error.npy')
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

    if True:#tracker==0:#error and error of error   last Navg_last trials:
        Navg_last=30
        lasterrs=np.average(Am[:,-Navg_last:],axis=1)
        lasterr=np.average(lasterrs)
        lasterrerr=np.std(lasterrs)/np.shape(lasterrs)[0]**.5
        print("lasterr=",lasterr, "+/-",lasterrerr)
        print("lasterrs",lasterrs)
        errorAm=Am
    yerr=np.std(Am,axis=0)/nrtries**.5
    aAm=np.average(Am,axis=0)

    aAms.append(aAm)
    yerrs.append(yerr)


plt.plot(aAms[0],color="blue",linewidth=.4,label="A=14")
plt.plot(aAms[1],color="red",linewidth=.4,label="A=10")
plt.plot(aAms[2],color="violet",linewidth=.4,label="A=14 and then 10")
linex=np.array([40,40])
liney=np.linspace(0.11,0.5,2)
plt.plot(linex,liney,color="red",markersize=.3)
plt.legend()
plt.show()



#df=pd.DataFrame({"A":aAms[0],"B":aAms[1]})
#df.A.plot(yerr=yerrs[0], capsize=2, style="o",markersize=10,linewidth=.5)
#df.B.plot(yerr=yerrs[1],secondary_y=True,capsize=2,color=(1,.5,0,.2))
#plt.legend()
#plt.plot()
sys.exit()

plt.figure()
for i in range(25):
    colidx=i/24
    plt.plot(errorAm[i],color=cm.rainbow(colidx),linewidth=1)
    plt.ylabel("error")
plitstring="lasterr"+str(round(lasterr,3))+"+-"+str(round(lasterrerr,3))
plt.savefig("bilder/plit_temp/the"+plitstring+".png")

plt.figure()
plt.plot(aAms[0],color="black",linewidth=.4)
plt.errorbar(x=range(len(aAms[0])),y=aAms[0], yerr=yerrs[0],color=(0,0,0,.2),elinewidth=.5)
plt.ylabel("error avg over "+str(nrtries)+" identical tries")
plt.savefig("bilder/plit_temp/this"+plitstring+".png")

plt.figure()
plt.plot(aAms[1],color="orange",linewidth=.4)
#plt.errorbar(x=range(len(aAms[1])),y=aAms[1], yerr=yerrs[1],color=(1,.5,0,.2),elinewidth=.5)
plt.ylabel("G avg over "+str(nrtries)+" identical tries")
plt.savefig("bilder/plit_temp/that"+plitstring+".png")
plt.figure()
plt.plot(aAms[2],color="green",linewidth=.4)
plt.errorbar(x=range(len(aAms[2])),y=aAms[2], yerr=yerrs[1],color=(1,.5,0,.2),elinewidth=.5)
plt.ylabel("eta_factor avg over "+str(nrtries)+" identical tries")
#plt.savefig("bilder/plit_temp/thot"+plitstring+".png")
plt.show()

