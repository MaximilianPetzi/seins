Nsims=10
allezehn=True
from ANNarchy import *
from reservoir_test import *
from kinematic import *
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

#3D stuff:
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

import importlib
import sys
import os
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.cm as cm


sim = 1#sys.argv[1]
num_goals=8#int(sys.argv[2])
num_trials = num_goals* 1000 #34 für pca e.g.
print("num_trials=",num_trials)

print("start sim=",sim)

setup(num_threads=2)

compile()

# initialize robot connection
sys.path.append('../../CPG_lib/MLMPCPG')
sys.path.append('../../CPG_lib/icubPlot')
iCubMotor = importlib.import_module(params.iCub_joint_names)
global All_Command
global All_Joints_Sensor
global myCont, angles, myT
All_Command = []
All_Joints_Sensor = []
RG_Layer_E = []
RG_Layer_F = []
PF_Layer_E = []
PF_Layer_F = []
MN_Layer_E = []
MN_Layer_F = []
myT = fSetTiming()

# Create list of CPG objects
myCont = fnewMLMPcpg(params.number_cpg)
# Instantiate the CPG list with iCub robot data
myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

"""
    NeckPitch, NeckRoll, NeckYaw, EyesTilt, EyesVersion, EyesVergence, TorsoYaw, TorsoRoll, TorsoPitch, RShoulderPitch, RShoulderRoll, \
    RShoulderYaw, RElbow, RWristProsup, RWristPitch, RWristYaw, RHandFinger, RThumbOppose, RThumbProximal, RThumbDistal, RIndexProximal, \
    RIndexDistal, RMiddleProximal, RMiddleDistal, RPinky, RHipPitch, RHipRoll, RHipYaw, RKnee, RAnklePitch, RAnkleRoll, \
    LShoulderPitch, LShoulderRoll, LShoulderYaw, LElbow, LWristProsup, LWristPitch, LWristYaw, LHandFinger, LThumbOppose, LThumbProximal, \
    LThumbDistal, LIndexProximal, LIndexDistal, LMiddleProximal, LMiddleDistal, LPinky, LHipPitch, LHipRoll, LHipYaw, LKnee, \
    LAnklePitch, LAnkleRoll
"""
# Initiate PF and RG patterns for the joints
# Initiate PF and RG patterns for the joints
joint1 = iCubMotor.LShoulderRoll
joint2 = iCubMotor.LElbow
joint3 = iCubMotor.LShoulderPitch
joint4 = iCubMotor.LShoulderYaw
joints = [joint4, joint3, joint1, joint2]
AllJointList = joints
num_joints = 4
angles = np.zeros(params.number_cpg)


angles[iCubMotor.LShoulderPitch] = 40
angles[iCubMotor.LElbow] = -10


# Update CPG initial position (reference position)
for i in range(0, len(myCont)):
    myCont[i].fUpdateInitPos(angles[i])
# Update all joints CPG, it is important to update all joints
# at least one time, otherwise, non used joints will be set to
# the default init position in the CPG which is 0
for i in range(0, len(myCont)):
    myCont[i].fUpdateLocomotionNetwork(myT, angles[i])




error_history = np.zeros(num_trials)
g_history=np.zeros(num_trials)
etaf_history=np.zeros(num_trials)
initial_position = wrist_position(np.radians(angles[joints]))[0:3]


def random_goal(initial_position):
    nvd = 0
    goal = [0, 0, 0]
    current_angles = np.copy(angles)
    while(nvd < 0.5):  # (nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.LShoulderPitch] = angles[iCubMotor.LShoulderPitch] + \
            np.random.normal(0, 20)
        current_angles[iCubMotor.LShoulderRoll] = angles[iCubMotor.LShoulderRoll] + \
            np.random.normal(0, 20)
        current_angles[iCubMotor.LShoulderYaw] = angles[iCubMotor.LShoulderYaw] + \
            np.random.normal(0, 20)
        current_angles[iCubMotor.LElbow] = angles[iCubMotor.LElbow] + \
            np.random.normal(0, 20)
        current_angles = np.radians(current_angles)
        goal = wrist_position(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
    return goal


def execute_movement(pms):
    myCont = fnewMLMPcpg(params.number_cpg)
    myCont = fSetCPGNet(myCont, params.my_iCub_limits,
                        params.positive_angle_dir)

    for j in range(4):
        myCont[joints[j]].fSetPatternRG(RG_Patterns(
            pms[j, 0], pms[j, 1], pms[j, 2], pms[j, 3]))
        myCont[joints[j]].fSetPatternPF(PF_Patterns(pms[j, 4], pms[j, 5]))

        myCont[joints[j]].RG.F.InjCurrent_value = 1.0 * \
            myCont[joints[j]].RG.F.InjCurrent_MultiplicationFactor
        myCont[joints[j]].RG.E.InjCurrent_value = -1.0 * \
            myCont[joints[j]].RG.E.InjCurrent_MultiplicationFactor

    current_angles = np.copy(angles)
    current_angles = np.radians(current_angles)

    # execute a movement
    for i in AllJointList:
        myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
    for idx, controller in enumerate(myCont):
        iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
    # iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
    All_Command.append(iCubMotor.MotorCommand[:])
    All_Joints_Sensor.append(current_angles)
    I = 0
    while I < 120:
        I += 1
        for i in AllJointList:
            myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
        for idx, controller in enumerate(myCont):
            iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
        # iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
        All_Command.append(iCubMotor.MotorCommand[:])
        All_Joints_Sensor.append(current_angles)

    mc_a = np.array(iCubMotor.MotorCommand[:])
    final_pos = wrist_position(mc_a[joints])[0:3]
    return final_pos

def record_net(t):
    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)

    inp[(t % num_goals)].r = 1.0
    simulate(200)
    inp.r = 0.0
    simulate(200)

    rec = m.get()
    return rec['r']

goal_history = np.zeros((num_goals, 3))
for i in range(num_goals):
    goal_history[i] = random_goal(initial_position)

g_growth=1  #1=neutral
#Wrec.effectvie_eta=1.0
whist=[]
rRhist=[]
noises=[]

for t in range(num_trials):
    print("trial: ",t)
    if os.path.isfile("stop"):#"touch stop" to manually stop all jiggling.py processes
        print("stop")
        sys.exit()
    np.save("lasttime.npy", time.time())   #update latest time
    #print("I'm still working!")
    #if t%100==0:
    #    print("trial", t)
    #print('trial '+str(t))
    current_goal = goal_history[t % num_goals]

    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)

    inp[(t % num_goals)].r = 1.0

    simulate(200)
    np.save("lasttime.npy", time.time())   #update latest time
    inp.r = 0.0
    #hier:
    simulate(200)

    rec = m.get()
    

    output = rec['r'][-200:, -24:]
    noises.append(rec["noise"])
    
    #hier:
    #plt.plot(rec['r'][:,12:19])
    #plt.show()
    dopca=True
    plotnow=False
    
    if allezehn==False:
        if t%(2*num_goals)==0:
            plotnow=True
    if dopca and (False or plotnow==True):
        pcamin=num_goals*6#num_trials-10*num_goals
        firstpcasample=pcamin-0*num_goals
        if True:#firstpcasample==t-1:
            pcaarray=np.zeros((400,0))
            coloridx=0
        #if firstpcasample<t and t<=pcamin:
        if t==pcamin:
            for pt in range(7):
                reco=record_net(t=t)
                np.concatenate((pcaarray,reco),axis=1)
                pca = PCA(n_components=3)
            #if t>pcamin and t%num_goals<2 and t%(num_goals*14)<2:
            
            
        if t>pcamin:
            fig=plt.figure()
            #3D:
            ax = fig.add_subplot(projection='3d')
            for pt in range(4):
                #coloridx=min(1,max(0,t/num_goals/300))
                #coloridx=(t%num_goals)/(num_goals-1)
                #simulate(2000)
                #rec = m.get()
                reco=record_net(t=t)#means that the goal is also fixed
                pcacomps=pca.fit_transform(reco)
                
                print("expl_var_ratio ",pca.explained_variance_ratio_)
                #plt.figure()######    color=(0,coloridx,1-coloridx,.4)   color=cm.rainbow(coloridx))
                #plt.subplot(2,1,1)
                
                #3D:
                #if t%num_goals==0:markersize=20
                #if t%num_goals==1:markersize=70
                ax.plot(pcacomps[:201,0],pcacomps[:201,1],pcacomps[:201,2],color=cm.rainbow(coloridx),linewidth=1.3)
                ax.plot(pcacomps[200:,0],pcacomps[200:,1],pcacomps[200:,2],color=cm.rainbow(coloridx),linewidth=.5,label=str(coloridx))
                ax.set_xlabel('component 1')
                ax.set_ylabel('component 2')
                ax.set_zlabel('component 3')
                print(coloridx,"=c, output=",np.mean(reco[-200:, -24:], axis=0))
                coloridx+=(1-coloridx)/4
            plt.legend()
            plt.savefig("bilder/sharedbasis_trial"+str(t/num_goals)+".png")
            if t>num_goals*40:plt.show();dskldsklsdlkfj=input("press anything")
            

    output = np.mean(output, axis=0)
    current_parms = np.zeros((4, 6))
    current_parms += output.reshape((4, 6))

    current_parms[:, 0] = np.clip((1+current_parms[:, 0])*(5.0/2.0), 0.001, 5)
    current_parms[:, 1] = np.clip((1+current_parms[:, 1])*(5.0/2.0), 0.001, 5)
    current_parms[:, 2] = np.clip(current_parms[:, 2]*4, -4, 4)
    current_parms[:, 3] = np.clip(
        (1+current_parms[:, 3])*(10.0/2.0), 0.001, 10)
    current_parms[:, 4] = np.clip((1+current_parms[:, 4]), 0.01, 2.0)
    current_parms[:, 5] = np.clip((1+current_parms[:, 5]), 0.01, 2.0)

    final_pos = execute_movement(current_parms)

    distance = np.linalg.norm(final_pos-current_goal)
    error = distance
    weightlist68=np.array(Wrec.w)
    weightlist69=np.reshape(weightlist68,(N**2))
    ghat=np.std(weightlist69)*np.sqrt(N)
    #manuel g steuern
    #print(ghat)
    #if (1+t)%(40*num_goals)==0:
    #    plt.plot(np.convolve(error_history,np.ones(8)))
    #    plt.plot(g_history)
    #    plt.show()
    #    g_growth=float(input("growth-factor=?"))
    #wrecc=np.array(Wrec.w)
    #wrecc*=g_growth
    #Wrec.w=(wrecc).tolist()
    rRhist_val=0
    if(t > 6*num_goals):
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[t % num_goals]
        Wrec.mean_mean_error = R_mean_mean[t % num_goals]
        rRhist_val=np.abs(Wrec.error-Wrec.mean_error)
        eta_lr=0.5
        #pop.A += -0.01-eta_lr*(Wrec.mean_error-Wrec.mean_mean_error)  #Wrec.effective_eta
        if pop.A<0:pop.A==0
        #if t%8==0:                                                                     
        #    print("A=",pop.A," (deltaA=",-0.01-eta_lr*(Wrec.mean_error-Wrec.mean_mean_error),") Rmean=",Wrec.mean_error)

        # Learn for one step
        step()
        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get()
    rRhist.append(rRhist_val)


    R_mean[t % num_goals] = alpha * R_mean[t %
                                           num_goals] + (1. - alpha) * error
    R_mean_mean[t % num_goals] = alpha * R_mean_mean[t %
                                           num_goals] + (1. - alpha) * R_mean[t % num_goals]
                                        
    error_history[t] = error
    g_history[t] = ghat
    whist.append(weightlist69[::111])
    #np.save("whist",whist)
    #plt.scatter(t*np.ones(len(weightlist69[:220:11])),weightlist69[:220:11],s=4)
    etaf_history[t]= pop.A#Wrec.effective_eta#*(R_mean[t % num_goals]-R_mean_mean[t % num_goals])
    #print(R_mean[t % num_goals]-R_mean_mean[t % num_goals],Wrec.eta)  

while True:
    try:
        content=np.load("paramfile.npy",allow_pickle=True)
        break
    except:
        time.sleep(0.1)
        
todo=content.item().get("todo")
done=content.item().get("done")

dirname=str(todo[0,0])+"_"+str(todo[0,1])+"_"+str(todo[0,2])+"_"+str(todo[0,3])

#np.save('error_org/'+dirname+"/"+sim+'error.npy', [error_history,g_history,etaf_history])

for gol in range(num_goals):
    if gol == 0:
        errh = np.zeros(len(error_history[gol:-num_goals:num_goals]))
        gh=np.zeros(len(g_history[gol:-num_goals:num_goals]))
        etafh=np.zeros(len(etaf_history[gol:-num_goals:num_goals]))
    if len(error_history) % num_goals==0:
        errh += error_history[gol:-num_goals:num_goals]
        gh += g_history[gol:-num_goals:num_goals]
        etafh += etaf_history[gol:-num_goals:num_goals]
    else:
        errh += error_history[gol:-(len(error_history) % num_goals):num_goals]
        gh += g_history[gol:-(len(g_history) % num_goals):num_goals]
        etafh += etaf_history[gol:-(len(etaf_history) % num_goals):num_goals]
errh /= num_goals
gh/= num_goals
etafh/=num_goals
#print("length of each errorhistory: ", len(errh))

#save file as the lowest free name (if some file is missing)
def sliding_avg(array,alpha):
    ret=[]
    for i in range(np.shape(array)[0]-alpha):
        ret.append(np.average(array[i:i+alpha],axis=0))
    return np.array(ret)

print("shapes errh, rRhist",np.shape(error_history),np.shape(rRhist))

#errh=sliding_avg(errh,5)
#rRhist=sliding_avg(rRhist,5)


np.save("noises.npy",{"noises":noises,"error_history":error_history})


sys.exit()
noisez=np.load("noises.npy",allow_pickle=True)
noises=noisez.item().get("noises")
errorhistory=noisez.item().get("error_history")
pertcount=[]
errdeltas=[]
print(len(noises),len(errorhistory))
for i in range(1,len(noises)):
    pertim=noises[i]
    pertim_bin=(pertim!=0)
    pert_count=np.sum(pertim_bin)
    pertcount.append(pert_count)
    errdeltas.append(errorhistory[i]-errorhistory[i-1])
plt.plot(pertcount);plt.plot(errdeltas);plt.show()
counts, bins = np.histogram(pertcount)

errarr=[]#this doesnt make any sense because errdeltas is pointless as there are 8 goals
binz=np.copy(bins)
binz[0]=-1
binz[-1]=100900800
for binnr in range(1,len(bins)):
    errarrsmall=[]
    for i in range(1,len(pertcount)):
        if pertcount[i]<binz[binnr] and pertcount[i]>binz[binnr-1]:
            errarrsmall.append(errdeltas[i])
    if errarrsmall==[]:
        errarr.append(-1)
    else:
        errarr.append(np.average(errarrsmall))
#    err_weights.append()
print(len(errarr),len(bins))
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

plt.plot(sliding_avg(error_history,5))
plt.plot(sliding_avg(np.abs(rRhist),5))
plt.show()
#np.save("temporary/3_04_22_5000.npy",errh)
sys.exit()
for i in range(Nsims):
    #check if filename exists, if not: save under the missing name
    fileex=os.path.isfile("error_org/"+dirname+"/"+str(i+1)+"error.npy")
    if not fileex:
        np.save('error_org/'+dirname+"/"+str(i+1)+'error.npy', [errh,gh,etafh])
        maxcount=np.load("maxcount.npy")
        maxcount=maxcount-1
        np.save("maxcount.npy",maxcount)
        break




#plitstr="goals"+num_goals
#np.save("error_h/plitstr",plitstr)
#plt.savefig("bilder/pca.png")
#plt.show()
print("jiggling finished")