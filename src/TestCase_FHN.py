#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import random
import math
import time
import utils
import optimization
import shutil

# Set the color cycle to the "Accent" palette
colors = plt.cm.tab20c.colors
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

# Class constraint for IC when estimated
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

save_train = 1    # save training

#%% Model parameters
t_max             = 100                                  # time horizon
dt                = 1                                    # real time step
layers            = 1                                    # number of hidden layers
neurons           = 8                                    # number of neurons of each layer 
dt_base           = 1                                    # rescaling factor 
variance_init     = 0.0001                               # initial variance of weights
t                 = np.arange(0, t_max+dt, dt)[None,:]   # time values
dt_num            = 0.1                                  # numerical time step
t_num             = np.arange(0, t_max, dt_num)[None, :] # numerical time values
num_latent_states = 2                                    # dimension of the latent space
num_latent_params = 0                                    # number of unknown parameters to be estimated online
num_input_var     = 1                                    # number of input variables
v_min             = -2.5                                 # minimum value of v
v_max             = 2.5                                  # maximum value of v
w_min             = -2.5                                 # minimum value of w
w_max             = 2.5                                  # maximum value of w
u_min             = 0.2                                  # minimum value of I_ext
u_max             = 1                                    # maximum value of I_ext
x_min = np.array([v_min, w_min])
x_max = np.array([v_max, w_max])

#%% Generating folders
folder = 'results' + str(neurons) + '_hlayers_' + str(layers) + '/'

if os.path.exists(folder) == False:
    os.mkdir(folder)
else:
    shutil.rmtree(folder)
    os.mkdir(folder)
folder_train = folder + 'train/'

#%% Define problem and normalization
problem = {
    "input_parameters": [],
    "input_signals": [
        { "name": "I_ext" },
    ],
    "output_fields": [
        { "name": "v" },
        { "name": "w" }
    ]
}

normalization = {
    'time': {
        'time_constant' : dt_base
    },

    'input_signals': {
        'I_ext': {'min': u_min, 'max' : u_max}
    },

    'output_fields': {
        'v': { 'min': v_min, "max": v_max },
        'w': { 'min': w_min, "max": w_max }
    }
}

#%% Importing/Generating dataset
# Parameters definition
theta  = 1
NTrain = 50
NTest  = 50

trajectories = []
inputs       = []
x0           = []
epsilon      = 1/12.5
a            = 0.7
b            = 0.8

for i in range(1,NTrain+1):

    Iext    = np.random.uniform(0.5,0.9)
    v0      = 1.5
    w0      = v0

    x_temp = utils.traj_computation(epsilon,a,b,Iext,theta,t_max,dt_num,v0,w0,0)
    trajectories.append(x_temp)
    x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

    u        = np.full((len(x_temp), 1), (2.0*Iext-u_min-u_max)/(u_max-u_min), dtype=np.float64)
    inputs.append(u)

x0_train        = np.stack(x0, axis=0).astype(np.float64)
u_train         = np.stack(inputs, axis=0).astype(np.float64)
training_target = np.stack(trajectories, axis=0).astype(np.float64)

#%%

trajectories = []
inputs       = []
x0           = []
a            = 0.7
b            = 0.8
epsilon      = 1/12.5

for i in range(1,NTest+1):
    
    Iext    = np.random.uniform(0.5,0.9)
    v0      = 1.5
    w0      = v0

    x_temp = utils.traj_computation(epsilon,a,b,Iext,theta,t_max,dt_num,v0,w0,0)
    trajectories.append(x_temp)
    x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

    u        = np.full((len(x_temp), 1), (2.0*Iext-u_min-u_max)/(u_max-u_min), dtype=np.float64)
    inputs.append(u)

x0_test        = np.stack(x0, axis=0).astype(np.float64)
u_test         = np.stack(inputs, axis=0).astype(np.float64)
testing_target = np.stack(trajectories, axis=0).astype(np.float64)

#%% Dataset parameters
n_size             = x0_train.shape[0]
n_size_testg       = x0_test.shape[0]
training_var_numpy = u_train
testing_var_numpy  = u_test

dataset_train = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': None,               # [num_samples x num_par]
        'inp_signals'   : training_var_numpy, # [num_samples x num_times x num_signals]
        'out_fields'    : training_target,    # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_testg = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': None,               # [num_samples x num_par]
        'inp_signals'   : testing_var_numpy,  # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target,     # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
#%%
np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi_real(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
print(dataset_train["inp_signals"].shape)
print(dataset_train["out_fields"].shape)

#%% Define model
input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation = tf.nn.tanh, input_shape = input_shape),#4
            tf.keras.layers.Dense(5, activation = tf.nn.tanh),#8
            tf.keras.layers.Dense(num_latent_states)
        ])
NNdyn.summary()

#%% Defining the forward evolution model for the latent transmission rate
def evolve_dynamics(dataset, initial_lat_state): #initial_state (n_samples x n_latent_state)
    
    lat_state = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - tf.constant(1)):
        inputs = [lat_state, dataset['inp_signals'][:,i,:]]
        lat_state = lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

#%% Loss functions
def loss_exp_beta(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    MSE = tf.reduce_mean(tf.square((state) - dataset['out_fields'])) #siccome è tutto normalizzato possiamo considerareMSE assoluto
    return MSE

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

#%% Loss weights
nu_loss_train = 1    #3e-2 # weight MSE metric
alpha_reg     = 1e-8       # regularization of trainable variables

#%% Training
trainable_variables_train = NNdyn.variables

def loss_train():
    l = nu_loss_train * loss_exp_beta(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_valid():
    l = loss_exp_beta(dataset_testg, x0_test)
    return l

def val_train():
    l = loss_exp_beta(dataset_train, x0_train)
    return l 

val_metric = loss_valid

#%% Training (Routine step 1)         
opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 500 #500
num_epochs_BFGS_train = 500 #1000

# Ho provato a fare Adam con riduzione del learning rate (si potrebbe provare una policy tipo reduce on plateau) + BFGS: risultato migliore è err generalizzazione circa 1e-7
print('training (Adam)...')
init_adam_time = time.time()
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-2))
end_adam_time = time.time()

print('training (Adam)...')
init_adam_time = time.time()
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-3))
end_adam_time = time.time()

print('training (Adam)...')
init_adam_time = time.time()
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-3))
end_adam_time = time.time()

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_train)
end_bfgs_time = time.time()

train_times = [end_adam_time - init_adam_time, end_bfgs_time - init_bfgs_time]

#%% Saving the rhs-NN (NNdyn) 
#NNdyn.save(folder + 'NNdyn')

#%% Testing
variables = evolve_dynamics(dataset_testg, x0_test)
variables = variables[17,:,:]
tt        = t_num[0,:]
target    = testing_target[17,:,:]

plt.plot(tt,5/2* variables[:,0],    'k--', label='v pred')
plt.plot(tt, target[:,0], 'r-', label='v true')
plt.plot(tt, 5/2*variables[:,1],    'b--', label='w pred')
plt.plot(tt, target[:,1], 'g-', label='w true')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('NeuralODE: Traiettoria vera vs predetta')
plt.legend()
plt.grid(True)
plt.savefig(folder + 'test1.png')

#%% Saving results
if os.path.exists(folder_train) == False:
    os.mkdir(folder_train)

if save_train:
    beta_train = evolve_dynamics(dataset_train, x0_train)
    l_train    = loss_exp_beta(dataset_train, x0_train)
    beta_testg = evolve_dynamics(dataset_testg, x0_test)
    l_testg    = loss_exp_beta(dataset_testg, x0_test)
    
    np.savetxt(folder_train + 'testg_error.txt', np.array(l_testg).reshape((1,1)))
    np.savetxt(folder_train + 'train_error.txt', np.array(l_train).reshape((1,1)))
    np.savetxt(folder_train + 'testg_train_error.txt', np.array(l_testg / l_train).reshape((1,1)))
    np.savetxt(folder_train + 'train_times.txt', train_times)
    np.savetxt(folder_train + 't_num.txt', tt)

os.system('cp TestCase_FHN.py ' + folder)

# Saving model
checkpoint = tf.train.Checkpoint(model_variables=trainable_variables_train)

checkpoint.save(folder + "variables_NNdyn")
# %%
