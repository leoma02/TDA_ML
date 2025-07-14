#!/usr/bin/env python3

#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_probability as tfp
import os
import sys
import random
import math
import time

# Set the color cycle to the "Accent" palette
colors = plt.cm.tab20c.colors
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization

# Class constraint for IC when estimated
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

#%% Set some hyperparameters
save_train        = 1    # save training

#%% Model parameters
t_max = 100 # time horizon

#%% Set numerical parameters
dt            = 0.1                    # real time step
layers        = 1                      # number of hidden layers
neurons       = 8                      # number of neurons of each layer 
dt_base       = 1                      # rescaling factor 
variance_init = 0.0001                 # initial variance of weights
t = np.arange(0, t_max+dt, dt)[None,:] # time values

dt_num = 0.1                                 # numerical time step
t_num = np.arange(0, t_max, dt_num)[None, :] # numerical time values

num_latent_states = 2    # dimension of the latent space
num_latent_params = 0    # number of unknown parameters to be estimated online
num_input_var     = 1    # number of input variables

#%% Generating folders
folder = '/home/leoma/Tesi/inferringTRdynamicsML/' + str(neurons) + '_hlayers_' + str(layers) + '/'

#size_num_v = 100 # random value, just to check if the code runs

if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'
folder_testg = folder + 'testg/'


problem = {
    "input_parameters": [],
    "input_signals": [
        { "name": "I_ext" }
    ],
    "output_fields": [
        { "name": "v" },
        { "name": "w" }
    ]
}
#%% Define problem
normalization = {
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'I_ext': {'min': 0.2,  'max' : 1},
    },
    'output_fields': {
        'v': { 'min': -2.5, "max": 2.5 },
        'w': { 'min': -2,   "max": 2 }
    }
}

#%% Importing dataset

# Parameters definition
theta  = 1
NTrain = 100
NTest  = 1

trajectories = []
inputs       = []
x0           = []

for i in range(1,NTrain+1):
  #epsilon = max(0.05,np.random.normal(1/12.5, 0.25))
  epsilon = 1/12.5
  #a       = np.random.normal(0.9,0.5)
  a       = 0.7
  #b       = np.random.normal(0.9,0.5)
  b       = 0.8
  Iext    = np.random.normal(0.5,0.2)
  #v0      = np.random.normal(1.5,5)
  v0      = 1.5
  w0      = v0

  x_temp = utils.traj_computation(epsilon,a,b,Iext,theta,t_max,dt_num,v0,w0,0)
  trajectories.append(x_temp)

  # x0 batch
  x0.append(x_temp[0])

  # u_seq batch costante
  u = np.full((len(x_temp), 1), Iext, dtype=np.float64)
  inputs.append(u)

x0_train        = np.stack(x0, axis=0).astype(np.float64)
u_train         = np.stack(inputs, axis=0).astype(np.float64)
training_target = np.stack(trajectories, axis=0).astype(np.float64)

trajectories = []
inputs       = []
x0           = []

for i in range(1,NTest+1):
  #epsilon = max(0.05,np.random.normal(1/12.5, 0.25))
  epsilon = 1/12.5
  #a       = np.random.normal(0.9,0.5)
  a       = 0.7
  #b       = np.random.normal(0.9,0.5)
  b       = 0.8
  Iext    = np.random.normal(0.5,0.2)
  #v0      = np.random.normal(1.5,5)
  v0      = 1.5
  w0      = v0

  x_temp = utils.traj_computation(epsilon,a,b,Iext,theta,t_max,dt_num,v0,w0,0)
  trajectories.append(x_temp)

  # x0 batch
  x0.append(x_temp[0])

  # u_seq batch costante
  u = np.full((len(x_temp), 1), Iext, dtype=np.float64)
  inputs.append(u)

x0_test        = np.stack(x0, axis=0).astype(np.float64)
u_test         = np.stack(inputs, axis=0).astype(np.float64)
testing_target = np.stack(trajectories, axis=0).astype(np.float64)

#%% Dataset parameters
n_size       = x0_train.shape[0]
n_size_testg = x0_test.shape[0]

# Generating datasets
training_var_numpy = u_train
testing_var_numpy  = u_test

dataset_train = {
        'times'         : t_num.T, # [num_times]
        'inp_parameters': None, # [num_samples x num_par]
        'inp_signals'   : training_var_numpy, # [num_samples x num_times x num_signals]
        'target'        : training_target, #  [num_samples x num_times x num_targets]
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_testg = {
        'times'         : t_num.T, # [num_times]
        'inp_parameters': None, # [num_samples x num_par]
        'inp_signals'   : testing_var_numpy, # [num_samples x num_times x num_signals]
        'target'        : testing_target, #  [num_samples x num_times x num_targets]
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

#%% Define model

# Initial conditions
InitialValue_train = 1.5*np.ones((dataset_train['num_samples'], 2))
InitialValue_testg = 1.5*np.ones((dataset_testg['num_samples'], 2))

input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

#%% Defining the neural network model
NNdyn = tf.keras.Sequential()
NNdyn.add(tf.keras.Input(shape=(input_shape)))
NNdyn.add(tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=variance_init)))
NNdyn.add(tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=variance_init)))
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
    MSE = tf.reduce_mean(tf.square((state) - dataset['target']) / tf.square(dataset['target'] )) 
    return MSE

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

#%% Loss weights
nu_loss_train = 3e-2 # weight MSE metric
nu_r1         = 5e-3 # weight regularization latent param
nu_p          = 1e-3 # weight \beta_threshold
nu_p2         = 1e-3 # weight H1 norm on \beta (1st point FD)
nu_p3         = 1e-6 # weight H1 norm on \beta
nu_s_inf      = 1e-7 # penalization of final susceptibles
alpha_reg     = 1e-8 # regularization of trainable variables

#%% Training
trainable_variables_train = NNdyn.variables

'''   GRANDE DUBBIO
def loss_train():
    beta = evolve_dynamics(dataset_train, x0_train)
    l = nu_loss_train *loss_exp_beta(dataset_train, beta) + alpha_reg * weights_reg(NNdyn)
    return l
'''

def loss_train():
    l = nu_loss_train *loss_exp_beta(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

''' GRANDE DUBBIO (ANALOGO)
# Validation metric
def val_train():
    beta = evolve_dynamics(dataset_train, x0_train)
    l = loss_exp_beta(dataset_train, beta)
    return l 
'''

def val_train():
    l = loss_exp_beta(dataset_train, x0_train)
    return l 

val_metric = val_train

#%% Training (Routine step 1) 
tf.config.run_functions_eagerly(True)
        
opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 30 #500
num_epochs_BFGS_train = 30 #1000

print('training (Adam)...')
init_adam_time = time.time()
#opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-3 * 25 / size_num_v))
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-3))
end_adam_time = time.time()

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_train)
end_bfgs_time = time.time()

train_times = [end_adam_time - init_adam_time, end_bfgs_time - init_bfgs_time]

tf.config.run_functions_eagerly(False)
#%% Saving the rhs-NN (NNdyn) 
#NNdyn.save(folder + 'NNdyn')

#%%plt.figure(figsize=(8,5))
variables = evolve_dynamics(dataset_testg, x0_test)
variables = variables[0,:,:]
tt        = t_num[0,:]
target    = testing_target[0,:,:]

plt.plot(tt, variables[:,0],    'k--', label='v pred')
plt.plot(tt, target[:,0], 'r-', label='v true')
plt.plot(tt, variables[:,1],    'b--', label='w pred')
plt.plot(tt, target[:,1], 'g-', label='w true')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('NeuralODE: Traiettoria vera vs predetta')
plt.legend()
plt.grid(True)
plt.show()

#%% Testing 

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

os.system('cp /home/leoma/Tesi/inferringTRdynamicsML/src/TestCase_FHN.py ' + folder)
# %%
