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
import TDA_utils

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

save_train      = 1    # save training
coarse_training = 0    # enable/disable training with coarse dataset
#%% 
####################
# MODEL PARAMETERS #
####################

t_max             = 100                                      # time horizon
t_max_ext         = 200                                      # extended time horizon
dt                = 1                                        # real time step
layers            = 1                                        # number of hidden layers
neurons           = 8                                        # number of neurons of each layer 
dt_base           = 1                                        # rescaling factor 
variance_init     = 0.0001                                   # initial variance of weights
t                 = np.arange(0, t_max+dt, dt)[None,:]       # time values
t_ext             = np.arange(0, t_max_ext+dt, dt)[None,:]   # extended time values
dt_num            = 1                                      # numerical time step
dt_ratio          = 50                                       # time step for coarse training data defined as dt_coarse/dt_num (for now, only multiples of dt_num)
t_num             = np.arange(0, t_max, dt_num)[None, :]     # numerical time values
t_num_ext         = np.arange(0, t_max_ext, dt_num)[None, :] # numerical time values
num_latent_states = 2                                        # dimension of the latent space
num_latent_params = 0                                        # number of unknown parameters to be estimated online
num_input_var     = 1                                        # number of input variables
v_min             = -2.5                                     # minimum value of v
v_max             = 2.5                                      # maximum value of v
w_min             = -2.5                                     # minimum value of w
w_max             = 2.5                                      # maximum value of w
u_min             = 0.2                                      # minimum value of I_ext
u_max             = 1.4                                      # maximum value of I_ext
a_min             = 0.1                                      # minimum value of a
a_max             = 1.0                                      # maximum value of a
b_min             = 0.1                                      # minimum value of b
b_max             = 1.0                                      # maximum value of b
epsilon_min       = 0.0                                      # minimum value of epsilon
epsilon_max       = 1.2                                      # maximum value of epsilon

#%%
######################
# GENERATING FOLDERS #
######################

folder = 'results' + str(neurons) + '_hlayers_' + str(layers) + '/'

shutil.rmtree(folder)

if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'

#%%
######################
# PROBLEM DEFINITION #
######################

problem = {
    "input_parameters": [
        { "name": "a" },
        { "name": "b" },
        { "name": "epsilon" },
    ],
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

    'input_parameters': {
        'a': {'min': a_min, 'max': a_max},
        'b': {'min': b_min, 'max': b_max},
        'epsilon': {'min': epsilon_min, 'max': epsilon_max}
    },

    'input_signals': {
        'I_ext': {'min': u_min, 'max' : u_max}
    },

    'output_fields': {
        'v': { 'min': v_min, "max": v_max },
        'w': { 'min': w_min, "max": w_max }
    }
}

#%%
#####################
# IMPORTING DATASET #
#####################

# Parameters definition
theta     = 1
NTrain    = 50 
NTest     = 50
NTest_ext = 50 

# Generating training dataset
x0_train, u_train, training_target, a_train, b_train, eps_train = utils.generate_dataset(NTrain, normalization, theta, t_max, dt_num)

# Generating testing dataset
x0_test, u_test, testing_target, a_test, b_test, eps_test = utils.generate_dataset(NTest, normalization, theta, t_max, dt_num)
testing_target = training_target
x0_test = x0_train
a_test = a_train
b_test = b_train
eps_test = eps_train

# Generating extended testing dataset
x0_test_ext, u_test_ext, testing_target_ext, a_test_ext, b_test_ext, eps_test_ext = utils.generate_dataset(NTest_ext, normalization, theta, t_max_ext, dt_num)

#%%
######################
# DATASET PARAMETERS #
######################

n_size                 = x0_train.shape[0]
n_size_testg           = x0_test.shape[0]
training_var_numpy     = u_train
testing_var_numpy      = u_test
testing_var_numpy_ext  = u_test_ext
inp_params_train       = np.stack((a_train[:,0,0],b_train[:,0,0],eps_train[:,0,0]), axis=-1)
inp_params_test        = np.stack((a_test[:,0,0],b_test[:,0,0],eps_test[:,0,0]), axis=-1)
inp_params_test_ext    = np.stack((a_test_ext[:,0,0],b_test_ext[:,0,0],eps_test_ext[:,0,0]), axis=-1)
coarse_indexes         = np.arange(0,len(t_num[0,:]),dt_ratio)

dataset_train = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': inp_params_train,   # [num_samples x num_par]
        'inp_signals'   : training_var_numpy, # [num_samples x num_times x num_signals]
        'out_fields'    : training_target,    # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_testg = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': inp_params_test,    # [num_samples x num_par]
        'inp_signals'   : testing_var_numpy,  # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target,     # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_test_ext = {
        'times'         : t_num_ext.T,           # [num_times]
        'inp_parameters': inp_params_test_ext,   # [num_samples x num_par]
        'inp_signals'   : testing_var_numpy_ext, # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target_ext,    # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max_ext,
        'time_vec'      : t_ext.T,
        'frac'          : int(dt/dt_num)
}
dataset_coarse = {
        'times'         : t_num.T,            # [num_times]
        'coarse_indexes': coarse_indexes,     
        'inp_parameters': inp_params_train,   # [num_samples x num_par]
        'inp_signals'   : training_var_numpy, # [num_samples x num_times x num_signals]
        'out_fields'    : training_target,    # [num_samples x num_times x num_targets] # RINOMINATO CAMPO
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
#%%
###################
# PROCESS DATASET #
###################

np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi_real(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_test_ext, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_coarse, problem, normalization, dt = None, num_points_subsample = None)
print(dataset_train["inp_signals"].shape)
print(dataset_train["out_fields"].shape)
dataset_testg = dataset_train

#%%
##############################
# NEURAL OPERATOR DEFINITION #
##############################

input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation = tf.nn.tanh, input_shape = input_shape),#4
            tf.keras.layers.Dense(5, activation = tf.nn.tanh),#8
            tf.keras.layers.Dense(num_latent_states)
        ])
NNdyn.summary()

#%%
###################
# EVOLUTION MODEL #
###################

def evolve_dynamics(dataset, initial_lat_state): #initial_state (n_samples x n_latent_state)
    
    lat_state = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    dt_ref = normalization['time']['time_constant']
    inp_params = dataset['inp_parameters']  # shape (N, num_params)
    
    # time integration
    for i in tf.range(dataset['num_times'] - tf.constant(1)):
        inputs = [lat_state, dataset['inp_signals'][:,i,:], inp_params]
        lat_state = lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

#%%
##################
# LOSS FUNCTIONS #
##################

def loss_MSE(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    MSE = tf.reduce_mean(tf.square((state) - dataset['out_fields'])) #siccome è tutto normalizzato possiamo considerareMSE assoluto
    return MSE

def loss_MSE_coarse(dataset, lat_states):
    state   = evolve_dynamics(dataset, lat_states)
    indexes = tf.convert_to_tensor(dataset['coarse_indexes'], dtype=tf.int32)

    state_sel = tf.gather(state, indexes, axis=1)
    out_sel   = tf.gather(dataset['out_fields'], indexes, axis=1)

    MSE = tf.reduce_mean(tf.square(state_sel - out_sel))
    return MSE

def loss_MSE_matrixnorm(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    MSE = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    matrix_loss = tf.zeros(state.shape[0], dtype=tf.float64)
    for i in range(0,state.shape[1],5):
        for j in range(0,i+1,step=5):
            #print("i:", i, "j:", j)
            d1 = (state[:,i,0] - state[:,j,0])**2 + (state[:,i,1] - state[:,j,1])**2
            d2 = (dataset['out_fields'][:,i,0] - dataset['out_fields'][:,j,0])**2 + (dataset['out_fields'][:,i,1] - dataset['out_fields'][:,j,1])**2
            diff = (d1 - d2)**2
            matrix_loss += diff
            #/(state.shape[1]/5)

    return MSE + 1e-4 * tf.reduce_mean(matrix_loss)

def loss_matrixnorm(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    
    matrix_loss = tf.zeros(state.shape[0], dtype=tf.float64)
    for i in range(0,state.shape[1],5):
        for j in range(0,i+1,step=5):
            #print("i:", i, "j:", j)
            d1 = (state[:,i,0] - state[:,j,0])**2 + (state[:,i,1] - state[:,j,1])**2
            d2 = (dataset['out_fields'][:,i,0] - dataset['out_fields'][:,j,0])**2 + (dataset['out_fields'][:,i,1] - dataset['out_fields'][:,j,1])**2
            diff = (d1 - d2)**2
            matrix_loss += diff
            #/(state.shape[1]/5)

    return tf.reduce_mean(matrix_loss)

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

#%%
################
# LOSS WEIGHTS #
################

nu_loss_train = 1    #3e-2 # weight MSE metric
alpha_reg     = 0#1e-8       # regularization of trainable variables

#%%
######################
# TRAINING FUNCTIONS #
######################

trainable_variables_train = NNdyn.variables

def loss_train():
    l = nu_loss_train * loss_MSE(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_train_matrixnorm():
    l = nu_loss_train * loss_MSE_matrixnorm(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    #l = nu_loss_train * loss_matrixnorm(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_valid():
    #l = loss_MSE(dataset_testg, x0_test)
    l = loss_MSE(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def val_train():
    l = loss_MSE(dataset_train, x0_train)
    return l 

def loss_valid_ext():
    l = loss_MSE(dataset_test_ext, x0_test_ext)
    return l

def loss_train_coarse():
    l = nu_loss_train * loss_MSE_coarse(dataset_coarse, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

val_metric = loss_valid

#%%
#######################
# NON COARSE TRAINING #
#######################

if coarse_training == 0:
    losses_dict = {'Standard': loss_train_matrixnorm, 'MatrixNorm': loss_train_matrixnorm} 
    opt_train   = optimization.OptimizationProblem(trainable_variables_train, losses_dict, val_metric)

    num_epochs_Adam_train        = 2000 #500
    num_epochs_BFGS_train        = 2000 #1000
    num_epochs_BFGS_matrix_train = 20# 2000

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

    variables1 = evolve_dynamics(dataset_testg, x0_test)
    num_plot  = 6
    #rand_vec  = np.random.randint(0,NTest,num_plot)
    rand_vec  = np.random.randint(0,NTrain,num_plot)
    tt        = t_num[0,:]

    fig, axs = plt.subplots(2,int(num_plot/2), figsize=(15,9))

    for i in range(2):
        for j in range(int(num_plot/2)):
            ind = rand_vec[2*i+j]
            axs[i,j].plot(tt, testing_target[ind,:,0], 'r-', label='v true')
            axs[i,j].plot(tt, 5/2*variables1[ind,:,0], 'k--', label='v pred')
            axs[i,j].plot(tt, testing_target[ind,:,1], 'g-', label='w true')
            axs[i,j].plot(tt, 5/2*variables1[ind,:,1], 'b--', label='w pred')
            axs[i,j].set_xlabel('Time')
            axs[i,j].set_ylabel('State')
            axs[i,j].set_title('NeuralODE: Traiettoria vera vs predetta')
            axs[i,j].grid(True)
            axs[i,j].legend(loc='upper right')

    plt.savefig(folder + 'test_prematrix.png')
'''
    opt_train.set_loss_train('MatrixNorm')

    print('training (BFGS)...')
    init_adam_time = time.time()
    opt_train.optimize_BFGS(num_epochs_BFGS_matrix_train)
    end_adam_time = time.time()

    variables2 = evolve_dynamics(dataset_testg, x0_test)
    fig, axs = plt.subplots(2,int(num_plot/2), figsize=(15,9))

    for i in range(2):
        for j in range(int(num_plot/2)):
            ind = rand_vec[2*i+j]
            axs[i,j].plot(tt, testing_target[ind,:,0], 'r-', label='v true')
            axs[i,j].plot(tt, 5/2*variables2[ind,:,0], 'k--', label='v pred')
            axs[i,j].plot(tt, testing_target[ind,:,1], 'g-', label='w true')
            axs[i,j].plot(tt, 5/2*variables2[ind,:,1], 'b--', label='w pred')
            axs[i,j].set_xlabel('Time')
            axs[i,j].set_ylabel('State')
            axs[i,j].set_title('NeuralODE: Traiettoria vera vs predetta')
            axs[i,j].grid(True)
            axs[i,j].legend(loc='upper right')

    plt.savefig(folder + 'test_postmatrix.png')

    train_times = [end_adam_time - init_adam_time, end_bfgs_time - init_bfgs_time]
'''

#%%
###################
# COARSE TRAINING #
###################

if coarse_training == 1:        
    opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train_coarse, val_metric)

    num_epochs_Adam_train        = 200 #500
    num_epochs_BFGS_train        = 200 #1000
    num_epochs_BFGS_matrix_train = 100

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

#%%
################
# MODEL SAVING #
################

#NNdyn.save(folder + 'NNdyn')

#%%
###################
# RESULTS SUMMARY #
###################

if coarse_training == 0:
    print('Training loss: ', loss_train().numpy())
if coarse_training == 1:
    print('Training loss: ', loss_train_coarse().numpy())
print('Training error: ', val_train().numpy())
print('Testing error: ', loss_valid().numpy())
print('Extended testing error: ', loss_valid_ext().numpy())

#%%
############################
# NON COARSE RESULTS PLOTS #
############################

if coarse_training == 0:
    variables = evolve_dynamics(dataset_train, x0_train)#evolve_dynamics(dataset_testg, x0_test)
    num_plot  = 6
    rand_vec  = np.array([0, 1, 2, 7, 8, 9])#np.random.randint(0,NTrain,num_plot)
    tt        = t_num[0,:]

    fig, axs = plt.subplots(2,int(num_plot/2), figsize=(15,9))

    for i in range(2):
        for j in range(int(num_plot/2)):
            ind = rand_vec[2*i+j]
            axs[i,j].plot(tt, testing_target[ind,:,0], 'r-', label='v true')
            axs[i,j].plot(tt, 5/2*variables[ind,:,0], 'k--', label='v pred')
            axs[i,j].plot(tt, testing_target[ind,:,1], 'g-', label='w true')
            axs[i,j].plot(tt, 5/2*variables[ind,:,1], 'b--', label='w pred')
            axs[i,j].set_xlabel('Time')
            axs[i,j].set_ylabel('State')
            axs[i,j].set_title('NeuralODE: Traiettoria vera vs predetta')
            axs[i,j].grid(True)
            axs[i,j].legend(loc='upper right')

    plt.savefig(folder + 'test.png')

#%%
########################
# COARSE RESULTS PLOTS #
########################

if coarse_training == 1:
    variables = evolve_dynamics(dataset_coarse, x0_test)
    num_plot  = 4
    rand_vec  = np.random.randint(0,50,num_plot)
    tt        = t_num[0,:]
    indexes   = dataset_coarse['coarse_indexes']

    fig, axs = plt.subplots(2,int(num_plot/2), figsize=(15,9))

    for i in range(2):
        for j in range(int(num_plot/2)):
            ind = rand_vec[2*i+j]
            axs[i,j].plot(tt[indexes], testing_target[ind,indexes,0], 'r-', label='v true (coarse)')
            #axs[i,j].plot(tt, testing_target[ind,:,0], 'y--', label='v true')
            axs[i,j].plot(tt, 5/2*variables[ind,:,0], 'k--', label='v pred')
            axs[i,j].plot(tt[indexes], testing_target[ind,indexes,1], 'g-', label='w true (coarse)')
            #axs[i,j].plot(tt, testing_target[ind,:,1], 'y-', label='w true')
            axs[i,j].plot(tt, 5/2*variables[ind,:,1], 'b--', label='w pred')
            axs[i,j].set_xlabel('Time')
            axs[i,j].set_ylabel('State')
            axs[i,j].set_title('NeuralODE: Traiettoria vera vs predetta')
            axs[i,j].grid(True)
            axs[i,j].legend(loc='upper right')

    plt.savefig(folder + 'test.png')

#%%
##########################
# EXTENDED RESULTS PLOTS #
##########################

variables = evolve_dynamics(dataset_test_ext, x0_test_ext)
num_plot  = 4
rand_vec  = np.random.randint(0,50,num_plot)
tt        = t_num[0,:]
tt_ext    = t_num_ext[0,:]
tt_comp   = np.setdiff1d(tt_ext, tt)

fig, axs = plt.subplots(2,int(num_plot/2), figsize=(15,9))

for i in range(2):
    for j in range(int(num_plot/2)):
        ind = rand_vec[2*i+j]
        axs[i,j].plot(tt, testing_target_ext[ind,:len(tt),0], 'r-', label='v true')
        axs[i,j].plot(tt_comp, testing_target_ext[ind,len(tt):,0], 'y-', label='v true ext', linewidth=2)
        axs[i,j].plot(tt_ext, 5/2*variables[ind,:,0], 'k--', label='v pred')
        axs[i,j].plot(tt, testing_target_ext[ind,:len(tt),1], 'g-', label='w true')
        axs[i,j].plot(tt_comp, testing_target_ext[ind,len(tt):,1], 'y-', label='w true ext', linewidth=2)
        axs[i,j].plot(tt_ext, 5/2*variables[ind,:,1], 'b--', label='w pred')
        axs[i,j].set_xlabel('Time')
        axs[i,j].set_ylabel('State')
        axs[i,j].set_title('NeuralODE: Traiettoria vera vs predetta')
        axs[i,j].grid(True)
        axs[i,j].legend(loc='upper right')

plt.savefig(folder + 'test_extended.png')
#%% Saving results
if os.path.exists(folder_train) == False:
    os.mkdir(folder_train)

if save_train:
    beta_train = evolve_dynamics(dataset_train, x0_train)
    l_train    = loss_MSE(dataset_train, x0_train)
    beta_testg = evolve_dynamics(dataset_testg, x0_test)
    l_testg    = loss_MSE(dataset_testg, x0_test)
    
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
