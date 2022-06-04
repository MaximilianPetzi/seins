import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
import matplotlib.cm as cm
#matplotlib.use("TkAgg")
import os
import sys

#change strlist (from line 25), plots (from line 100)
#       

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

#(on coarse grid)

#eta3_A:
#strlist=["36.0_0.4_3.0_800.0/","22.0_0.4_3.0_800.0/","14.0_0.4_3.0_800.0/","8.0_0.4_3.0_800.0/","4.0_0.4_3.0_800.0/"]
#eta3_f:
#strlist=["22.0_0.4_3.0_800.0/","22.0_1.0_3.0_800.0/","22.0_3.0_3.0_800.0/","22.0_8.0_3.0_800.0/","22.0_18.0_3.0_800.0/"]
#eta3_eta:  
#strlist=["22.0_0.4_3.0_800.0/","22.0_0.4_1.0_800.0/","22.0_0.4_0.3_800.0/","22.0_0.4_0.1_800.0/","22.0_0.4_0.02_800.0/"]

#(on fine grid)
#eta.5_A:
#strlist=["90.0_0.15_0.5_800.0/","60.0_0.15_0.5_800.0/","42.0_0.15_0.5_800.0/","28.0_0.15_0.5_800.0/","18.0_0.15_0.5_800.0/","10.0_0.15_0.5_800.0/"]
#eta.5_f:
#strlist=["42.0_0.15_0.5_800.0/","42.0_0.25_0.5_800.0/","42.0_0.7_0.5_800.0/","42.0_2.0_0.5_800.0/"]
#eta.5_eta:
#strlist=["42.0_0.15_6.0_800.0/","42.0_0.15_0.5_800.0/","42.0_0.15_0.01_800.0/"]

#for g:
#strlist=["22.0_0.4_3.0_800.0/"]
strlist=["22.0_1.0_3.0_800.0/"]

#for 5000:
#strlist=["3_04_22_5000.npy/"]   #also change line 57, 19 and 61,62,63
for tracker in range(len(strlist)):
    am = []
    am2=[]
    for i in range(nrt):
        #print(i+1)
        try:
            #old:
            ##trackers = np.load("varA_errors/6_goals/"+strlist[tracker]+str(i+1)+'error.npy')#trackers = np.load("error_h/"+str(i+1)+'error.npy')
            #normal:
            trackers = np.load("collection/three_grid/all/"+strlist[tracker]+str(i+1)+'error.npy')#trackers = np.load("error_h/"+str(i+1)+'error.npy')
            #for 5000:
            #trackers = np.load("collection/3_04_22_5000.npy")
            #trackers = np.load("error_h/"+str(i+1)+'error.npy')
            print("-____________-\n",trackers,np.shape(trackers))
            print("here:"+"collection/three_grid/all/"+strlist[tracker]+str(i+1)+'error.npy')
            if tracker==7:print("grey: ",np.shape(trackers))
            t=trackers[0]
            t2=trackers[1]
            am.append(t[:990])#990
            am2.append(t2[:990])
        except:
            pass
            print("missing: "+"collection/three_grid/all/"+strlist[tracker]+str(i+1)+'error.npy')

            #print("missing: varA_errors/"+strlist[tracker]+str(i+1)+'error.npy')
    am=np.array(am)
    am2=np.array(am2)
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
    
    #aAm=np.average(Am[:18],axis=0)
    
    aAms.append(Am)
    yerrs.append(yerr)
#print("shapes aAms[i]:",np.shape(aAms[0]),np.shape(aAms[1]),np.shape(aAms[2]),np.shape(aAms[3]),np.shape(aAms[4]))

swindow=1
print(np.shape(aAms))
aAms=np.array(aAms)#this is the error
aAms_unaveraged=aAms[0]
aAms=np.average(aAms,axis=1)
#for coarse grid:
print(np.shape(aAms))

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="A=36")
#plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="A=22")
#plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="A=14")

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="f=0.4")
#plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="f=1")
#plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="f=3")
#plt.plot(sliding_avg(aAms[3],swindow),color="orange",linewidth=.6,label="f=8")
#plt.plot(sliding_avg(aAms[4],swindow),color="red",linewidth=.6,label="f=18")

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="$\eta$=3.0")
#plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="$\eta$=1.0")
##plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="$\eta$=0.3")
##plt.plot(sliding_avg(aAms[3],swindow),color="orange",linewidth=.6,label="$\eta$=0.1")
##plt.plot(sliding_avg(aAms[4],swindow),color="red",linewidth=0.6,label="$\eta$=0.02")

###########################################################################################
#for fine grid:
#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.5,label="A=90")
#plt.plot(sliding_avg(aAms[1],swindow),color="blue",linewidth=.5,label="A=60")
#plt.plot(sliding_avg(aAms[2],swindow),color="green",linewidth=.5,label="A=42")
##plt.plot(sliding_avg(aAms[3],swindow),color="yellow",linewidth=.9,label="A=28")
##plt.plot(sliding_avg(aAms[4],swindow),color="orange",linewidth=.5,label="A=18")
##plt.plot(sliding_avg(aAms[5],swindow),color="red",linewidth=.5,label="A=10")

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="f=0.15")
#plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="f=0.25")
#plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="f=0.7")
#plt.plot(sliding_avg(aAms[3],swindow),color="orange",linewidth=.6,label="f=2")

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.6,label="$\eta$=6.0")
#plt.plot(sliding_avg(aAms[1],swindow),color="green",linewidth=.6,label="$\eta$=0.5")
#plt.plot(sliding_avg(aAms[2],swindow),color="yellow",linewidth=.9,label="$\eta$=0.01")

#aAmss_avg=np.average(aAmss,axis=0)
aAm2=am2    #this is g
#aAm2=np.average(am,axis=0)

#to plot g graphs:
#plt.plot(sliding_avg(aAm2[0],swindow),color="black",linestyle="-.",linewidth=1)
#plt.plot(sliding_avg(aAm2[1],swindow),color="blue",linestyle="-.",linewidth=1)
#plt.plot(sliding_avg(aAm2[2],swindow),color="green",linestyle="-.",linewidth=1)
#plt.plot(sliding_avg(aAm2[3],swindow),color="yellow",linestyle="-.",linewidth=1.3)
#plt.plot(sliding_avg(aAm2[4],swindow),color="orange",linestyle="-.",linewidth=1)
#plt.plot(sliding_avg(aAm2[5],swindow),color="red",linestyle="-.",linewidth=1)
#matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20)
#to plot errors graphs for the  graphs
plt.plot(sliding_avg(aAms_unaveraged[0],swindow),color="black",linewidth=.4)
plt.plot(sliding_avg(aAms_unaveraged[1],swindow),color="blue",linewidth=.4)
plt.plot(sliding_avg(aAms_unaveraged[2],swindow),color="green",linewidth=.4)
plt.plot(sliding_avg(aAms_unaveraged[3],swindow),color="yellow",linewidth=.7)
plt.plot(sliding_avg(aAms_unaveraged[4],swindow),color="orange",linewidth=.4)
plt.plot(sliding_avg(aAms_unaveraged[5],swindow),color="red",linewidth=.4)

#for 5000:

#plt.plot(sliding_avg(aAms[0],swindow),color="black",linewidth=.4)

plt.xlabel("trials")
plt.ylabel("error")
plt.title("6 simulations with bad settings ($\eta$=3, A=22, f=1)")
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


leg = plt.legend()

for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

#plt.ylabel("error avg over 25 identical tries")
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
