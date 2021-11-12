N=800
from ANNarchy import *
import matplotlib.pyplot as plt
import os.path
from scipy.stats import multivariate_normal
import pickle

neuron = Neuron(
    #hier
    parameters = """
        tau = 30.0 : population # Time constant
        constant = 0.0 # The four first neurons have constant rates
        alpha = 0.05 : population # To compute the sliding mean 0.05
        f = 9.0 : population # Frequency of the perturbation 3/9
        A = 20. : population # Perturbation amplitude. dt*A/tau should be 0.5... original=16/20
    """,
    equations="""
        # Perturbation
        perturbation = if Uniform(0.0, 1.0) < f/1000.: 1.0 else: 0.0 
        noise = if perturbation > 0.5: A*Uniform(-1.0, 1.0) else: 0.0

        # ODE for x
        x += dt*(sum(in) + sum(exc) - x + noise)/tau

        # Output r
        rprev = r
        r = if constant == 0.0: tanh(x) else: tanh(constant)

        # Sliding mean
        delta_x = x - x_mean
        x_mean = alpha * x_mean + (1 - alpha) * x
    """
)

#changed weight change to positive
#eta * trace * (mean_error) * (error - mean_error)
synapse = Synapse(
    parameters="""
        #eta = 1.0: projection # Learning rate 0.5 -- 0.6 in icubs_bg/2
        learning_phase = 0.0 : projection # Flag to allow learning only at the end of a trial
        error = 0.0 : projection # Reward received
        mean_error = 0.0 : projection # Mean Reward received
        mean_mean_error = 0.0 : projection
        max_weight_change = 0.0005 : projection # Clip the weight changes 0.0003/0.0005sss
    """,
    equations="""
        # Trace
        trace += if learning_phase < 0.5:
                    power(pre.rprev * (post.delta_x), 3)
                 else:
                    0.0

        # effective_eta = if learning_phase > 0.5:
        #     if -eta*(mean_error-mean_mean_error)>eta*0.1:
        #     -eta*(mean_error-mean_mean_error)
        #     else:
        #         eta*0.1
        #     else:
        #         0.5*eta:projection
        eta_lr=0.0:projection
        eta += if learning_phase > 0.5:
            -eta_lr*(mean_error-mean_mean_error)
            else:
                0.0:projection

        effective_eta = if learning_phase > 0.5:
            eta
            else:
                0.0:projection


        # Weight update only at the end of the trial
        delta_w = if learning_phase > 0.5:
                effective_eta * trace * (mean_error) * (error - mean_error)
             else:
                 0.0 : min=-max_weight_change, max=max_weight_change
        w -= if learning_phase > 0.5:
                delta_w
             else:
                 0.0
    """
)

inp = Population(9, Neuron(parameters="r=0.0"))
inp.r = 0.0



# Recurrent population

pop = Population(N, neuron)
pop[1].constant = 1.0
pop[10].constant = 1.0
pop[11].constant = -1.0
pop.x = Uniform(-0.1, 0.1)

# Input weights
Wi = Projection(inp, pop, 'in')
#Wi.connect_from_file(filename='Wi.data')
Wi.connect_all_to_all(weights=Uniform(-0.2, 0.2))
#Wi.connect_fixed_number_post(number=300,weights=Uniform(-0.5, 0.5))


# Recurrent weights
#hier
g = 1.0 #1.0
Wrec = Projection(pop,pop,'exc',synapse)  #pop[0:(N-28)], pop, 'exc', synapse)
#Wrec.connect_from_file(filename='Wrec.data')
Wrec.connect_all_to_all(weights=Normal(0., g/np.sqrt(N)), allow_self_connections=True)

#Wrec2 = Projection(pop[-28:],pop[0:N-28], 'exc', synapse)
#Wrec2.connect_all_to_all(weights=Normal(0., g/np.sqrt(N)), allow_self_connections=True)

#compile()


# Compute the mean reward per trial
R_mean = np.zeros(100)
R_mean_mean=np.zeros(100)
alpha = 0.33 #0.75 0.33 
alpha2=0.01

m = Monitor(pop,['r'])
#mp = Monitor(pop,['r'])
#mi = Monitor(inp_e,['r'])
#mp = Monitor(pop,['r'])
#mie = Monitor(inp_e,['r'])
#mi = Monitor(inp,['r'])
