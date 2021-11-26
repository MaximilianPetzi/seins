
from ANNarchy import *
from reservoir import *
from kinematic import *
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

import importlib
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.cm as cm

sim = sys.argv[1]
num_goals=int(sys.argv[2])
num_trials = num_goals* 150 #34 für pca e.g.
print("num_trials=",num_trials)

print(sim)

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


goal_history = np.zeros((num_goals, 3))
for i in range(num_goals):
    goal_history[i] = random_goal(initial_position)

g_growth=1
#Wrec.effectvie_eta=1.0

for t in range(num_trials):
    print('trial '+str(t))
    current_goal = goal_history[t % num_goals]

    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)

    inp[(t % num_goals)].r = 1.0

    simulate(200)

    inp.r = 0.0
    #hier:
    simulate(200)

    rec = m.get()
    

    output = rec['r'][-200:, -24:]

    
    #hier:
    #plt.plot(rec['r'][:,12:19])
    #plt.show()
    dopca=False
    if dopca:
        pcamin=num_goals*12#num_trials-10*num_goals
        firstpcasample=pcamin-5*num_goals
        if firstpcasample==t-1:
            pcaarray=np.zeros((400,0))
            coloridx=0
        if firstpcasample<t and t<=pcamin:
            print(np.shape(pcaarray),np.shape(rec["r"]))
            np.concatenate((pcaarray,rec['r']),axis=1)
        if t==pcamin:
            pca = PCA(n_components=2)
            print("datashape=",np.shape(pcaarray))
        if t>pcamin and t%num_goals<2 and t%(num_goals*14)<2:
            #coloridx=min(1,max(0,t/num_goals/300))
            #coloridx=(t%num_goals)/(num_goals-1)
            #simulate(2000)
            #rec = m.get()
            print("trial nr",t/num_goals)
            
            pcacomps=pca.fit_transform(rec['r'])
            
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)
            print(np.shape(pcacomps))
            #plt.figure()######    color=(0,coloridx,1-coloridx,.4)   color=cm.rainbow(coloridx))
            #plt.subplot(2,1,1)
            if t%num_goals==0:markersize=20
            if t%num_goals==1:markersize=70
            
            plt.scatter([pcacomps[0,0]],[pcacomps[0,1]],s=markersize,color=cm.rainbow(coloridx))
            plt.plot(pcacomps[:200,0],pcacomps[:200,1],color=cm.rainbow(coloridx),linewidth=.2,label="goal"+str(t%num_goals)+"during input")
            plt.plot(pcacomps[200:,0],pcacomps[200:,1],color=cm.rainbow(coloridx),linewidth=.8,label="goal"+str(t%num_goals)+"after input")
            plt.scatter([pcacomps[-1,0]],[pcacomps[-1,1]],marker='^',s=markersize,color=cm.rainbow(coloridx))
            #plt.text(pcacomps[0,0],pcacomps[0,1], str((t-t%num_goals)/num_goals), color="black", fontsize=12)
            #plt.text(pcacomps[-1,0],pcacomps[-1,1], str((t-t%num_goals)/num_goals), color="black", fontsize=12)
            #plt.legend()
            coloridx+=(1-coloridx)/8
            
            
            plt.xlabel("1st component")
            plt.ylabel("2nd component")
            plt.title("from trial nr. "+str(pcamin))
            #plt.subplot(2,1,2)
            #plt.plot(pca.explained_variance_ratio_)
            #plt.xlabel("Component number")
            #plt.ylabel("Explained variance")
            #plt.show()


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

    if(t > 6*num_goals):
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[t % num_goals]
        Wrec.mean_mean_error = R_mean_mean[t % num_goals]
        
        eta_lr=0
        #if t==num_goals*30:
        #    Wrec.effective_eta=0.25
        Wrec.effective_eta += -eta_lr*(Wrec.mean_error-Wrec.mean_mean_error)
        # Learn for one step
        step()
        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get()


    R_mean[t % num_goals] = alpha * R_mean[t %
                                           num_goals] + (1. - alpha) * error
    R_mean_mean[t % num_goals] = alpha * R_mean_mean[t %
                                           num_goals] + (1. - alpha) * R_mean[t % num_goals]
                                        
    error_history[t] = error
    g_history[t] = ghat
    etaf_history[t]= Wrec.effective_eta#*(R_mean[t % num_goals]-R_mean_mean[t % num_goals])
    #print(R_mean[t % num_goals]-R_mean_mean[t % num_goals],Wrec.eta)  
print(np.shape(error_history))
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
print("length of each errorhistory: ", len(errh))
np.save('error_h/'+sim+'error.npy', [errh,gh,etafh])

#plitstr="goals"+num_goals
#np.save("error_h/plitstr",plitstr)
print("last line:")
plt.savefig("bilder/pca.png")
#plt.show()