import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
import matplotlib.cm as cm
#matplotlib.use("TkAgg")
import os
import sys
#import pandas as pd
def sliding_avg(array,alpha):
    ret=[]
    for i in range(np.shape(array)[0]-alpha):
        ret.append(np.average(array[i:i+alpha],axis=0))
    return np.array(ret)
os.system("rm bilder/plit_temp/*")
nrt=10
aAms=[]
yerrs=[]
#strlist=["error_h_A14/","error_h_A10/","error_h_A14-A10after40/","after80/","after160/","secondA14/","after20_after400_7/","thirdA14/","fourthA14/"]
#strlist=["error_h_A14/","error_h_A10/","4_goals/A14/","4_goals/A10/","2_goals/A14/","2_goals/A10/"]
#strlist=["A20/","A14/","A12/","A10/"]
strlist=["10.0_9.0_1.0_800.0/","12.0_9.0_0.8_800.0/","14.0_9.0_0.6_800.0/","16.0_9.0_0.4_800.0/","18.0_9.0_0.2_800.0/","20.0_9.0_0.1_800.0/"]
#strlist=["16.0_3.0_1.0_800.0/","18.0_2.0_1.0_800.0/","20.0_1.5_1.0_800.0/"]
for tracker in range(len(strlist)):
    am = []
    for i in range(nrt):
        #print(i+1)
        try:
            #trackers = np.load("varA_errors/6_goals/"+strlist[tracker]+str(i+1)+'error.npy')#trackers = np.load("error_h/"+str(i+1)+'error.npy')
            trackers = np.load("collection/sA_20/"+strlist[tracker]+str(i+1)+'error.npy')#trackers = np.load("error_h/"+str(i+1)+'error.npy')
            print("here:"+strlist[tracker]+str(i+1)+'error.npy')
            if tracker==7:print("grey: ",np.shape(trackers))
            t=trackers[0]
            am.append(t[:990])
        except:
            pass
            print("missing: "+strlist[tracker]+str(i+1)+'error.npy')

            #print("missing: varA_errors/"+strlist[tracker]+str(i+1)+'error.npy')
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
        #lasterrs=np.average(Am[:,-Navg_last:],axis=1)
        #lasterr=np.average(lasterrs)
        #lasterrerr=np.std(lasterrs)/np.shape(lasterrs)[0]**.5
        #print("lasterr=",lasterr, "+/-",lasterrerr)
        #print("lasterrs",lasterrs)
        errorAm=Am
    yerr=np.std(Am,axis=0)/nrtries**.5
    aAm=np.average(Am[:18],axis=0)

    aAms.append(aAm)
    yerrs.append(yerr)
#print("shapes aAms[i]:",np.shape(aAms[0]),np.shape(aAms[1]),np.shape(aAms[2]),np.shape(aAms[3]),np.shape(aAms[4]))

swindow=4
plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="A")
plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="AA")
plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="AAA")
plt.plot(sliding_avg(aAms[3],swindow),color="orange",linewidth=.6,label="AAAA")
plt.plot(sliding_avg(aAms[4],swindow),color="red",linewidth=0.6,label="5A")
plt.plot(sliding_avg(aAms[5],swindow),color="violet",linewidth=.6,label="6A")

#plt.plot(sliding_avg(aAms[5],8),color="grey",linewidth=.6,label="2 goals, A10")
#plt.plot(sliding_avg(aAms[6],8),color="brown",linewidth=.6,label="after20_after400_7")
#plt.plot(sliding_avg(aAms[7],8),color="grey",linewidth=.6,label="A=14(3)")
#plt.plot(sliding_avg(aAms[8],8),color="grey",linewidth=.8,label="A=14(4)")

#plt.errorbar(x=range(len(aAms[3])),y=aAms[3], yerr=yerrs[3],color=(1,.5,0,.2),elinewidth=.5)
linex=np.array([40,40])
liney=np.linspace(0.11,0.5,2)
#plt.plot(linex,liney,color="blue",markersize=.3)
linex=np.array([80,80])
liney=np.linspace(0.11,0.5,2)
#plt.plot(linex,liney,color="violet",markersize=.3)
linex=np.array([160,160])
liney=np.linspace(0.11,0.5,2)
#plt.plot(linex,liney,color="red",markersize=.3)
plt.legend()
plt.ylabel("error avg over 25 identical tries")
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

