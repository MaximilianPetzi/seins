from ANNarchy import *
from reservoir_bg3 import *
from kinematic import *
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

import importlib
import sys
import time
import numpy as np

sim = sys.argv[1]
print(sim)

setup(num_threads=2)

compile()



#initialize robot connection
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
joints = [joint4,joint3,joint1,joint2]
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


num_goals = 2

num_trials = num_goals*300 #600

error_history = np.zeros(num_trials)



initial_position = wrist_position(np.radians(angles[joints]))[0:3]



def random_goal(initial_position):
    nvd = 0
    goal = [0,0,0]
    current_angles = np.copy(angles)
    while(nvd<0.5):#(nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.LShoulderPitch] = angles[iCubMotor.LShoulderPitch] + np.random.normal(0,20)
        current_angles[iCubMotor.LShoulderRoll] = angles[iCubMotor.LShoulderRoll] + np.random.normal(0,20)
        current_angles[iCubMotor.LShoulderYaw] = angles[iCubMotor.LShoulderYaw] + np.random.normal(0,20)
        current_angles[iCubMotor.LElbow] = angles[iCubMotor.LElbow] + np.random.normal(0,20)
        current_angles = np.radians(current_angles)
        goal = wrist_position(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
    return goal


def execute_movement(pms):
    myCont = fnewMLMPcpg(params.number_cpg)
    myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

    for j in range(4):
        myCont[joints[j]].fSetPatternRG(RG_Patterns(pms[j,0], pms[j,1], pms[j,2], pms[j,3]))
        myCont[joints[j]].fSetPatternPF(PF_Patterns(pms[j,4], pms[j,5]))

        myCont[joints[j]].RG.F.InjCurrent_value = 1.0 * myCont[joints[j]].RG.F.InjCurrent_MultiplicationFactor
        myCont[joints[j]].RG.E.InjCurrent_value = -1.0 * myCont[joints[j]].RG.E.InjCurrent_MultiplicationFactor

    current_angles = np.copy(angles)
    current_angles = np.radians(current_angles)

    #execute a movement
    for i in AllJointList:
            myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
    for idx, controller in enumerate(myCont):
            iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
    #iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
    All_Command.append(iCubMotor.MotorCommand[:])
    All_Joints_Sensor.append(current_angles)
    I=0
    while I<120:
        I+=1
        for i in AllJointList:
            myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
        for idx, controller in enumerate(myCont):
            iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
        #iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
        All_Command.append(iCubMotor.MotorCommand[:])
        All_Joints_Sensor.append(current_angles)

    mc_a = np.array(iCubMotor.MotorCommand[:])
    final_pos = wrist_position(mc_a[joints])[0:3]
    return final_pos


goal_history = np.zeros((num_goals,3))
for i in range(num_goals):
    goal_history[i] = random_goal(initial_position)


for t in range(num_trials):
    print('trial '+str(t))
    current_goal =  goal_history[t%num_goals]   



    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)

    inp[(t%num_goals)].r = 1.0
    
    simulate(200)

    inp.r = 0.0

    simulate(200)

    rec = m.get()
    
    output = rec['r'][-200:,-24:]
    output = np.mean(output,axis=0)


    current_parms = np.zeros((4,6))
    current_parms+=output.reshape((4,6))
    
    current_parms[:,0] = np.clip( (1+current_parms[:,0])*(5.0/2.0),0.001,5)  
    current_parms[:,1] = np.clip( (1+current_parms[:,1])*(5.0/2.0),0.001,5) 
    current_parms[:,2] = np.clip(current_parms[:,2]*4,-4,4)  
    current_parms[:,3] = np.clip( (1+current_parms[:,3])*(10.0/2.0),0.001,10)  
    current_parms[:,4] = np.clip( (1+current_parms[:,4]),0.01,2.0)  
    current_parms[:,5] = np.clip( (1+current_parms[:,5]),0.01,2.0) 

    final_pos = execute_movement(current_parms)


    distance = np.linalg.norm(final_pos-current_goal)
    error = distance 

    if(t>10):
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[t%num_goals]
        # Learn for one step
        step()
        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get()

    
    R_mean[t%num_goals] = alpha * R_mean[t%num_goals] + (1.- alpha) * error


    

    error_history[t] = error


np.save(sim+'error.npy',error_history)


