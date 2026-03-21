#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
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
#%% 
####################
# MODEL PARAMETERS #
####################

t_max             = 30                                       # time horizon
t_max_ext         = 200                                      # extended time horizon
dt                = 1                                        # real time step
layers            = 1                                        # number of hidden layers
neurons           = 8                                        # number of neurons of each layer 
dt_base           = 1                                        # rescaling factor 
variance_init     = 0.01                                   # initial variance of weights
t                 = np.arange(0, t_max+dt, dt)[None,:]       # time values
t_ext             = np.arange(0, t_max_ext+dt, dt)[None,:]   # extended time values
dt_num            = 0.15                                        # numerical time step
dt_test           = dt_num/2
dt_ratio          = 10                                       # time step for coarse training data defined as dt_coarse/dt_num (for now, only multiples of dt_num)
t_num             = np.arange(0, t_max, dt_num)[None, :]     # numerical time values
t_num_fine        = np.arange(0, t_max, dt_test)[None, :]    # numerical time values for fine testing
t_num_ext         = np.arange(0, t_max_ext, dt_num)[None, :] # numerical time values
num_latent_states = 2                                        # dimension of the latent space
num_latent_params = 0                                        # number of unknown parameters to be estimated online
num_input_var     = 1                                        # number of input variables
v_min             = -2.5                                     # minimum value of v
v_max             = 2.5                                      # maximum value of v
w_min             = -2.5                                     # minimum value of w
w_max             = 2.5                                      # maximum value of w
upper_limit       = 5
mi_min            = 0.0                                      # minimum value of mi
mi_max            = upper_limit                              # maximum value of mi

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
        { "name": "mi" },
    ],
    "input_signals": [
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
        'mi': {'min': mi_min, 'max': mi_max},
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
NTrain    = 500 
NTest     = 500
NTest_ext = 50 

# Generating training dataset
seed = 0
x0_train, training_target, mi_train = utils.generate_dataset(NTrain, normalization, t_max, dt_num, seed)

# Generating testing dataset
seed = 10
x0_test, testing_target, mi_test = utils.generate_dataset(NTest, normalization, t_max, dt_num, seed)

# Generating testing dataset
seed = 10
x0_test_fine, testing_target_fine, mi_test_fine = utils.generate_dataset(NTest, normalization, t_max, dt_test, seed)

# Generating extended testing dataset
seed = 20
x0_test_ext, testing_target_ext, mi_test_ext = utils.generate_dataset(NTest_ext, normalization, t_max_ext, dt_num, seed)

#%%
######################
# DATASET PARAMETERS #
######################

n_size                 = x0_train.shape[0]
n_size_testg           = x0_test.shape[0]
training_var_numpy     = None
testing_var_numpy      = None
testing_var_numpy_fine = None
testing_var_numpy_ext  = None
inp_params_train       = mi_train[:,0,:]
inp_params_test        = mi_test[:,0,:]
inp_params_test_fine   = mi_test_fine[:,0,:]
inp_params_test_ext    = mi_test_ext[:,0,:]
coarse_indexes         = np.arange(0,len(t_num[0,:]),dt_ratio)

dataset_train = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': inp_params_train,   # [num_samples x num_par]
        'inp_signals'   : None,               # [num_samples x num_times x num_signals]
        'out_fields'    : training_target,    # [num_samples x num_times x num_targets]
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_testg = {
        'times'         : t_num.T,            # [num_times]
        'inp_parameters': inp_params_test,    # [num_samples x num_par]
        'inp_signals'   : None,               # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target,     # [num_samples x num_times x num_targets]
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_num)
}
dataset_test_fine = {
        'times'         : t_num_fine.T,            # [num_times]
        'inp_parameters': inp_params_test_fine,    # [num_samples x num_par]
        'inp_signals'   : None,                    # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target_fine,     # [num_samples x num_times x num_targets]
        'num_times'     : t_max,
        'time_vec'      : t.T,
        'frac'          : int(dt/dt_test)
}
dataset_test_ext = {
        'times'         : t_num_ext.T,           # [num_times]
        'inp_parameters': inp_params_test_ext,   # [num_samples x num_par]
        'inp_signals'   : None,                  # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target_ext,    # [num_samples x num_times x num_targets]
        'num_times'     : t_max_ext,
        'time_vec'      : t_ext.T,
        'frac'          : int(dt/dt_num)
}
dataset_coarse = {
        'times'         : t_num.T,            # [num_times]
        'coarse_indexes': coarse_indexes,     
        'inp_parameters': inp_params_train,   # [num_samples x num_par]
        'inp_signals'   : None,               # [num_samples x num_times x num_signals]
        'out_fields'    : training_target,    # [num_samples x num_times x num_targets]
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
utils.process_dataset_epi_real(dataset_test_fine, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_test_ext, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_coarse, problem, normalization, dt = None, num_points_subsample = None)
print(dataset_train["inp_parameters"].shape)
print(dataset_train["out_fields"].shape)

# Extract topological information from dataset
# topological_indexes, topological_triangles = utils.extract_indexes(dataset_train['out_fields'])
# test_indexes, test_triangles               = utils.extract_indexes(dataset_testg['out_fields'])
# test_indexes_fine, test_triangles_fine     = utils.extract_indexes(dataset_test_fine['out_fields'])   
# pairs = np.zeros((NTrain, 2), dtype=np.int64)
# mask  = np.zeros((NTrain,), dtype=np.float64)
# for b, item in enumerate(topological_indexes):
#     if len(item) == 2:
#         pairs[b, 0] = int(item[0])
#         pairs[b, 1] = int(item[1])
#         mask[b] = 1.0
#     else:
#         pairs[b, 0] = 0
#         pairs[b, 1] = 0
#         mask[b] = 0.0
# pairs = tf.constant(pairs, tf.int64)
# mask = tf.constant(mask, tf.float64)

# Extract topological information from sw embedded dataset
# sw_v_train, sw_w_train = utils.sliding_window(dataset_train['out_fields'], window_size=2, delay=5, step=1)
# sw_v_test, sw_w_test   = utils.sliding_window(dataset_testg['out_fields'], window_size=2, delay=5, step=1)
# topological_indexes_v, topological_triangles_v = utils.extract_indexes(sw_v_train)
# topological_indexes_w, topological_triangles_w = utils.extract_indexes(sw_w_train)
# test_indexes_v, test_triangles_v               = utils.extract_indexes(sw_v_test)
# test_indexes_w, test_triangles_w               = utils.extract_indexes(sw_w_test)
# pairs_v = np.zeros((NTrain, 2), dtype=np.int64)
# pairs_w = np.zeros((NTrain, 2), dtype=np.int64)
# mask_v  = np.zeros((NTrain,), dtype=np.float64)
# mask_w  = np.zeros((NTrain,), dtype=np.float64)

# for b, item in enumerate(topological_indexes_v):
#     if len(item) == 2:
#         pairs_v[b, 0] = int(item[0])
#         pairs_v[b, 1] = int(item[1])
#         mask_v[b] = 1.0
#     else:
#         pairs_v[b, 0] = 0
#         pairs_v[b, 1] = 0
#         mask_v[b] = 0.0
# for b, item in enumerate(topological_indexes_w):
#     if len(item) == 2:
#         pairs_w[b, 0] = int(item[0])
#         pairs_w[b, 1] = int(item[1])
#         mask_w[b] = 1.0
#     else:
#         pairs_w[b, 0] = 0
#         pairs_w[b, 1] = 0
#         mask_w[b] = 0.0
# pairs_v = tf.constant(pairs_v, tf.int64)
# mask_v = tf.constant(mask_v, tf.float64)
# pairs_w = tf.constant(pairs_w, tf.int64)
# mask_w = tf.constant(mask_w, tf.float64)

# # Plot persistence triangles of dataset to check if algorithm is working properly
# for i in range(25):
#     fig, ax = plt.subplots(1,2,figsize=(10, 5))
#     ax[0].plot(np.arange(0,t_max,dt_num), training_target[i,:,0], 'r-', label='v true')
#     ax[0].plot(np.arange(0,t_max,dt_num), training_target[i,:,1], 'g-', label='w true')
#     ax[0].grid(True)
#     ax[1].scatter(dataset_train['out_fields'][i,:,0],dataset_train['out_fields'][i,:,1], label='predicted')
#     ind1 = topological_triangles[i][0]
#     ind2 = topological_triangles[i][1]
#     ind3 = topological_triangles[i][2]
#     ax[1].plot([dataset_train['out_fields'][i,ind1,0],dataset_train['out_fields'][i,ind2,0]],[dataset_train['out_fields'][i,ind1,1],dataset_train['out_fields'][i,ind2,1]])
#     ax[1].plot([dataset_train['out_fields'][i,ind1,0],dataset_train['out_fields'][i,ind3,0]],[dataset_train['out_fields'][i,ind1,1],dataset_train['out_fields'][i,ind3,1]])
#     ax[1].plot([dataset_train['out_fields'][i,ind2,0],dataset_train['out_fields'][i,ind3,0]],[dataset_train['out_fields'][i,ind2,1],dataset_train['out_fields'][i,ind3,1]])
#     ax[1].plot([dataset_train['out_fields'][i,topological_indexes[i][0],0],dataset_train['out_fields'][i,topological_indexes[i][1],0]],[dataset_train['out_fields'][i,topological_indexes[i][0],1],dataset_train['out_fields'][i,topological_indexes[i][1],1]], 'r--', label='largest persistence')
#     ax[1].legend()
#     ax[1].set_title('Phase Space')
#     ax[1].axis('equal')

#%%
##############################
# NEURAL OPERATOR DEFINITION #
##############################

input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

E_net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation=tf.nn.tanh, input_shape=input_shape),
    tf.keras.layers.Dense(8, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

D_net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation=tf.nn.tanh, input_shape=input_shape),
    tf.keras.layers.Dense(8, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.softplus)
])

weightsE = E_net.get_weights()
with open("init_E.pkl", "wb") as f:
    pickle.dump(weightsE, f)

weightsD = D_net.get_weights()
with open("init_D.pkl", "wb") as f:
    pickle.dump(weightsD, f)

# with open("init_E.pkl", "rb") as f:
#     good_init = pickle.load(f)
# E_net.set_weights(good_init)

# with open("init_D.pkl", "rb") as f:
#     good_init = pickle.load(f)
# D_net.set_weights(good_init)

def energy_and_grad(lat_state, inp_params):
    with tf.GradientTape() as tape:
        tape.watch(lat_state)
        z = tf.concat([lat_state, inp_params], axis=-1)
        E = E_net(z)
        E_sum = tf.reduce_sum(E)

    grad_E = tape.gradient(E_sum, lat_state)
    return E, grad_E

def structured_vector_field(lat_state, inp_params):
    lat_state = tf.convert_to_tensor(lat_state, dtype=tf.float64)
    inp_params = tf.convert_to_tensor(inp_params, dtype=tf.float64)
    _, grad_E = energy_and_grad(lat_state, inp_params)

    z = tf.concat([lat_state, inp_params], axis=-1)
    D = D_net(z)

    dE_dv = grad_E[:, 0:1]
    dE_dw = grad_E[:, 1:2]

    dv = dE_dw
    dw = -dE_dv - D * dE_dw

    return tf.concat([dv, dw], axis=-1), grad_E, D

#%%
###################
# EVOLUTION MODEL #
###################

def evolve_dynamics(dataset, initial_lat_state): #initial_state (n_samples x n_latent_state)
    
    lat_state = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    dt_ref     = normalization['time']['time_constant']
    inp_params = dataset['inp_parameters']  # shape (N, num_params)
    dt_int     = t_max/dataset['num_times']
    
    # time integration
    for i in tf.range(dataset['num_times'] - tf.constant(1)):
        inputs = [lat_state, inp_params]
        f = structured_vector_field(lat_state, inp_params)
        lat_state = lat_state + dt_int/dt_ref * f
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

def evolve_dynamics_symplectic(dataset, initial_lat_state):
    lat_state         = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size=dataset['num_times'])
    grad_trajectories = tf.TensorArray(tf.float64, size=dataset['num_times'])
    diss_trajectories = tf.TensorArray(tf.float64, size=dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    
    inp_params = dataset['inp_parameters']  # (N, num_params)
    dt         = t_max/dataset['num_times']
    dt         = dt / normalization['time']['time_constant']

    for i in tf.range(dataset['num_times'] - 1):
        v = lat_state[:, 0:1]
        w = lat_state[:, 1:2]

        f_n, grad_temp, D_temp = structured_vector_field(lat_state, inp_params)
        dw_n  = f_n[:, 1:2]
        w_new = w + dt * dw_n

        lat_mixed = tf.concat([v, w_new], axis=-1)
        f_m, grad_E, D = structured_vector_field(lat_mixed, inp_params)
        dv_m      = f_m[:, 0:1]
        v_new     = v + dt * dv_m

        lat_state         = tf.concat([v_new, w_new], axis=-1)
        lat_state_history = lat_state_history.write(i + 1, lat_state)
        grad_trajectories = grad_trajectories.write(i, grad_E)
        diss_trajectories = diss_trajectories.write(i, D)

    return tf.transpose(lat_state_history.stack(), perm=(1, 0, 2)), tf.transpose(grad_trajectories.stack(), perm=(1, 0, 2)), tf.transpose(diss_trajectories.stack(), perm=(1, 0, 2))

#%%
################
# LOSS WEIGHTS #
################

nu_loss_train = 1    # weight MSE metric
alpha_reg     = 1e-6 # regularization of trainable variables
lambda_1      = 1e-2    # weight MSE metric
lambda_2      = 3e-4 # regularization of energy gradient
lambda_3      = 3e-4 # regularization of dissipation term

#%%
##################
# LOSS FUNCTIONS #
##################

def loss_MSE(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))
    return MSE

def loss_MSE_symplectic(dataset, lat_states):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))
    return MSE

def loss_MSE_sw(dataset, lat_states, window_size, delay, step):
    state                = evolve_dynamics_symplectic(dataset, lat_states)
    sw_v, sw_w           = utils.sliding_window(state, window_size, delay, step)
    sw_v_true, sw_w_true = utils.sliding_window(dataset['out_fields'], window_size, delay, step)
    return tf.reduce_mean(tf.square(sw_v - sw_v_true)) + tf.reduce_mean(tf.square(sw_w - sw_w_true))

def loss_MSE_matrixnorm(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    MSE = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    matrix_loss = tf.zeros(state.shape[0], dtype=tf.float64)
    for i in range(0,state.shape[1],5):
        for j in range(0,i+1,step=5):
            d1 = (state[:,i,0] - state[:,j,0])**2 + (state[:,i,1] - state[:,j,1])**2
            d2 = (dataset['out_fields'][:,i,0] - dataset['out_fields'][:,j,0])**2 + (dataset['out_fields'][:,i,1] - dataset['out_fields'][:,j,1])**2
            diff = (d1 - d2)**2
            matrix_loss += diff

    return MSE + 1e-2 *tf.reduce_mean(matrix_loss)  / (state.shape[1]/5)**2

def loss_matrixnorm(dataset, lat_states):
    state = evolve_dynamics(dataset, lat_states)
    
    matrix_loss = tf.zeros(state.shape[0], dtype=tf.float64)
    for i in range(0,state.shape[1],5):
        for j in range(0,i+1,step=5):
            d1 = (state[:,i,0] - state[:,j,0])**2 + (state[:,i,1] - state[:,j,1])**2
            d2 = (dataset['out_fields'][:,i,0] - dataset['out_fields'][:,j,0])**2 + (dataset['out_fields'][:,i,1] - dataset['out_fields'][:,j,1])**2
            diff = (d1 - d2)**2
            matrix_loss += diff

    return  tf.reduce_mean(matrix_loss) / (state.shape[1]/5)**2

def MSE_topological_loss_vec_symplectic(dataset, lat_states, pairs, mask):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    sw_v, sw_w           = utils.sliding_window(state, 2, 5, 1)
    sw_v_true, sw_w_true = utils.sliding_window(dataset['out_fields'], 2, 5, 1)
    MSE                  = tf.reduce_mean(tf.square(sw_v - sw_v_true)) + tf.reduce_mean(tf.square(sw_w - sw_w_true))

    i = pairs[:, 0]  # (B,)
    j = pairs[:, 1]  # (B,)
    si = tf.gather(state, i, axis=1, batch_dims=1)  # (B,2)
    sj = tf.gather(state, j, axis=1, batch_dims=1)  # (B,2)
    ti = tf.gather(dataset['out_fields'],  i, axis=1, batch_dims=1)  # (B,2)
    tj = tf.gather(dataset['out_fields'],  j, axis=1, batch_dims=1)  # (B,2)

    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)

    per_traj = tf.square(d1 - d2)                 # (B,)

    masked = per_traj * mask

    denom  = tf.reduce_sum(mask) + 1e-10
    return MSE + tf.reduce_sum(masked) / denom

def MSE_topological_loss_sw(dataset, lat_states, pairs_v, pairs_w, mask_v, mask_w, window_size, delay, step):
    state      = evolve_dynamics_symplectic(dataset, lat_states)
    sw_v, sw_w = utils.sliding_window(state, window_size, delay, step)
    sw_v_true, sw_w_true = utils.sliding_window(dataset['out_fields'], window_size, delay, step)
    MSE        = tf.reduce_mean(tf.square(sw_v - sw_v_true)) + tf.reduce_mean(tf.square(sw_w - sw_w_true))

    i  = pairs_v[:, 0]  # (B,)
    j  = pairs_v[:, 1]  # (B,)
    si = tf.gather(sw_v, i, axis=1, batch_dims=1)  # (B,2)
    sj = tf.gather(sw_v, j, axis=1, batch_dims=1)  # (B,2)
    ti = tf.gather(sw_v_true,  i, axis=1, batch_dims=1)  # (B,2)
    tj = tf.gather(sw_v_true,  j, axis=1, batch_dims=1)  # (B,2)
    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)
    tv = tf.square(d1 - d2)                 # (B,)
    masked_v = tv * mask_v
    denom_v  = tf.reduce_sum(mask_v) + 1e-10

    i  = pairs_w[:, 0]  # (B,)
    j  = pairs_w[:, 1]  # (B,)
    si = tf.gather(sw_w, i, axis=1, batch_dims=1)  # (B,2)
    sj = tf.gather(sw_w, j, axis=1, batch_dims=1)  # (B,2)
    ti = tf.gather(sw_w_true,  i, axis=1, batch_dims=1)  # (B,2)
    tj = tf.gather(sw_w_true,  j, axis=1, batch_dims=1)  # (B,2)
    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)
    tw = tf.square(d1 - d2)                 # (B,)
    masked_w = tw * mask_w
    denom_w  = tf.reduce_sum(mask_w) + 1e-10
    return MSE + tf.reduce_sum(masked_v) / denom_v + tf.reduce_sum(masked_w) / denom_w

def MSE_topological_loss_symplectic(dataset, lat_states, indexes):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    topological_term = 0.0

    for k in range(state.shape[0]):
        if len(indexes[k]) != 0:
            i  = indexes[k][0]
            j  = indexes[k][1]
            d1 = tf.sqrt((state[k,i,0] - state[k,j,0])**2 + (state[k,i,1] - state[k,j,1])**2 + 1e-10)
            d2 = tf.sqrt((dataset['out_fields'][k,i,0] - dataset['out_fields'][k,j,0])**2 + (dataset['out_fields'][k,i,1] - dataset['out_fields'][k,j,1])**2 + 1e-10)

            topological_term += (d1-d2)**2

    return MSE + topological_term

def MSE_topological_loss_vec(dataset, lat_states, pairs, mask):
    state = evolve_dynamics(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    i = pairs[:, 0]  # (B,)
    j = pairs[:, 1]  # (B,)
    si = tf.gather(state, i, axis=1, batch_dims=1)  # (B,2)
    sj = tf.gather(state, j, axis=1, batch_dims=1)  # (B,2)
    ti = tf.gather(dataset['out_fields'],  i, axis=1, batch_dims=1)  # (B,2)
    tj = tf.gather(dataset['out_fields'],  j, axis=1, batch_dims=1)  # (B,2)
    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)

    per_traj = tf.square(d1 - d2)                 # (B,)

    masked = per_traj * mask

    denom  = tf.reduce_sum(mask) + 1e-10
    return MSE + tf.reduce_sum(masked) / denom

def extract_indexes_np(state_np):
    train_indexes, triangles = utils.extract_indexes(state_np)
    B = state_np.shape[0]
    pairs = np.zeros((B,2), dtype=np.int64)
    mask  = np.zeros((B,), dtype=np.float64)
    for k, it in enumerate(train_indexes):
        if it is not None and len(it)==2:
            pairs[k] = (int(it[0]), int(it[1]))
            mask[k]  = 1.0
        else:
            pairs[k] = (0,0)
            mask[k]  = 0.0
    return pairs, mask

def MSE_topological_loss(dataset, lat_states, indexes):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    topological_term         = 0.0

    for k in range(state.shape[0]):
        if len(indexes[k]) != 0:
            i  = indexes[k][0]
            j  = indexes[k][1]
            d1 = tf.sqrt((state[k,i,0] - state[k,j,0])**2 + (state[k,i,1] - state[k,j,1])**2 + 1e-10)
            d2 = tf.sqrt((dataset['out_fields'][k,i,0] - dataset['out_fields'][k,j,0])**2 + (dataset['out_fields'][k,i,1] - dataset['out_fields'][k,j,1])**2 + 1e-10)

            topological_term += (d1-d2)**2

    return MSE + topological_term

def MSE_bitopological_loss_symplectic_vec(dataset, lat_states, pairs_h1, mask_h1):

    state = evolve_dynamics_symplectic(dataset, lat_states)
    true  = dataset['out_fields']
    MSE   = tf.reduce_mean(tf.square(state - true))

    def _extract_pairs_mask_from_state(state_np):
        train_indexes, triangles = utils.extract_indexes(state_np)
        B                        = state_np.shape[0]
        pairs                    = np.zeros((B, 2), dtype=np.int64)
        mask                     = np.zeros((B,), dtype=np.float64)
        for k, it in enumerate(train_indexes):
            if it is not None and len(it) == 2:
                pairs[k, 0] = int(it[0])
                pairs[k, 1] = int(it[1])
                mask[k]     = 1.0
            else:
                pairs[k, :] = (0, 0)
                mask[k]     = 0.0
        return pairs, mask

    pairs_train, mask_train = tf.numpy_function(
        func=_extract_pairs_mask_from_state,
        inp=[state],
        Tout=[tf.int64, tf.float64]
    )

    pairs_train.set_shape([None, 2])
    mask_train.set_shape([None])

    i = pairs_h1[:, 0]
    j = pairs_h1[:, 1]
    l = pairs_train[:, 0]
    m = pairs_train[:, 1]

    si = tf.gather(state, i, axis=1, batch_dims=1)
    sj = tf.gather(state, j, axis=1, batch_dims=1)
    ti = tf.gather(true,  i, axis=1, batch_dims=1)
    tj = tf.gather(true,  j, axis=1, batch_dims=1)

    sl = tf.gather(state, l, axis=1, batch_dims=1)
    sm = tf.gather(state, m, axis=1, batch_dims=1)
    tl = tf.gather(true,  l, axis=1, batch_dims=1)
    tm = tf.gather(true,  m, axis=1, batch_dims=1)

    eps = tf.cast(1e-10, state.dtype)

    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1) + eps)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1) + eps)
    d3 = tf.sqrt(tf.reduce_sum(tf.square(sl - sm), axis=-1) + eps)
    d4 = tf.sqrt(tf.reduce_sum(tf.square(tl - tm), axis=-1) + eps)

    per_traj = tf.square(d1 - d2) + tf.square(d3 - d4)

    per_traj = per_traj * mask_h1
    denom    = tf.reduce_sum(mask_h1) + 1e-10

    topological_term = tf.reduce_sum(per_traj)

    return MSE + topological_term / denom

def weights_reg(E_net, D_net):
    S_E = sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in E_net.layers])/len(E_net.layers)
    S_D = sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in D_net.layers])/len(D_net.layers)
    return S_E + S_D

def structured_loss(dataset, lat_states):
    t,g,d = evolve_dynamics_symplectic(dataset,lat_states)

    return lambda_1*tf.reduce_mean(tf.square(t - dataset['out_fields'])) + lambda_2*tf.reduce_mean(tf.square(g)) + lambda_3*tf.reduce_mean(tf.square(d))

def structured_loss_sw(dataset, lat_states, window_size, delay, step):
    t,g,d = evolve_dynamics_symplectic(dataset,lat_states)

    sw_v, sw_w           = utils.sliding_window(t, window_size, delay, step)
    sw_v_true, sw_w_true = utils.sliding_window(dataset['out_fields'], window_size, delay, step)
    MSE                  = tf.reduce_mean(tf.square(sw_v - sw_v_true)) + tf.reduce_mean(tf.square(sw_w - sw_w_true))
    
    return lambda_1*MSE + lambda_2*tf.reduce_mean(tf.square(g)) + lambda_3*tf.reduce_mean(tf.square(d))

#%%
######################
# TRAINING FUNCTIONS #
######################

trainable_variables_train =E_net.trainable_variables + D_net.trainable_variables

def loss_train():
    l = nu_loss_train * loss_MSE(dataset_train, x0_train) + alpha_reg * weights_reg(E_net, D_net)
    return l

def loss_train_symplectic():
    l = nu_loss_train * loss_MSE_symplectic(dataset_train, x0_train) + alpha_reg * weights_reg(E_net, D_net)
    return l

def loss_train_sw():
    l = nu_loss_train * loss_MSE_sw(dataset_train, x0_train, window_size=2, delay=5, step=1) + alpha_reg * weights_reg(E_net, D_net)
    return l

def loss_train_matrixnorm():
    l = nu_loss_train * loss_MSE_matrixnorm(dataset_train, x0_train) + alpha_reg * weights_reg(E_net, D_net)
    #l = nu_loss_train * loss_matrixnorm(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_train_structured():
    l = structured_loss(dataset_train, x0_train) + alpha_reg * weights_reg(E_net, D_net)
    return l

def loss_train_structured_sw():
    l = structured_loss_sw(dataset_train, x0_train, window_size=2, delay=5, step=1) + alpha_reg * weights_reg(E_net, D_net)
    return l

# def loss_train_topological():
#     l = nu_loss_train * MSE_bitopological_loss_symplectic_vec(dataset_train, x0_train, pairs, mask) + alpha_reg * weights_reg(NNdyn)
#     #l = nu_loss_train * MSE_topological_loss_vec_symplectic(dataset_train, x0_train, pairs, mask) + alpha_reg * weights_reg(NNdyn)
#     #l = nu_loss_train * MSE_topological_loss_sw(dataset_train, x0_train, pairs_v, pairs_w, mask_v, mask_w, 2, 5, 1) + alpha_reg * weights_reg(NNdyn)
#     return l

def loss_valid():
    # l1 = loss_MSE(dataset_testg, x0_test)
    # l2 = loss_matrixnorm(dataset_testg, x0_test)
    # l1 = MSE_topological_loss_symplectic(dataset_train, x0_train, topological_indexes)
    # l2 = MSE_topological_loss_vec_symplectic(dataset_train, x0_train, pairs, mask)
    # l1 = MSE_topological_loss(dataset_train, x0_train, topological_indexes)
    # l2 = prova(dataset_train, x0_train, pairs, mask, 0.0)
    return 0,0

def loss_valid_MatrixNorm():
    l = loss_MSE_matrixnorm(dataset_testg, x0_test)
    return l

def val_train():
    l = loss_MSE(dataset_train, x0_train)
    return l 

def loss_valid_ext():
    l = loss_MSE(dataset_test_ext, x0_test_ext)
    return l

val_metric = loss_valid

#%%
#######################
# NON COARSE TRAINING #
#######################

f = open(folder +'training_data.txt','x')
losses_dict = {'Standard': loss_train_structured, 'TopoLoss': loss_train_sw} 
opt_train   = optimization.OptimizationProblem(trainable_variables_train, losses_dict, val_metric)

num_epochs_Adam_train        = 2500 #500
num_epochs_BFGS_train        = 2000 #1000
num_epochs_BFGS_matrix_train = 3000 #2000

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

#opt_train.set_loss_train('TopoLoss')

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_matrix_train)
end_bfgs_time = time.time()

#opt_train.set_loss_train('TopoLoss')

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_matrix_train)
end_bfgs_time = time.time()

tt         = t_num[0,:]
num_plot   = 6
rand_vec   = [0,1,2,3,4,5] #np.random.randint(0,NTest,num_plot)
variables3, _, _ = evolve_dynamics_symplectic(dataset_testg, x0_test)
fig, axs   = plt.subplots(2,int(num_plot/2), figsize=(15,9))

train_times = [end_adam_time - init_adam_time, end_bfgs_time - init_bfgs_time]

#%%
#####################
# TRAJECTORIES TEST #
#####################
variables3, _, _ = evolve_dynamics_symplectic(dataset_testg, x0_test)
for i in range(0,NTest,25):
    fig, ax = plt.subplots()
    ax.plot(tt, testing_target[i,:,0], 'r-', label='v true')
    ax.plot(tt, 5/2*variables3[i,:,0], 'k--', label='v pred')
    ax.plot(tt, testing_target[i,:,1], 'g-', label='w true')
    ax.plot(tt, 5/2*variables3[i,:,1], 'b--', label='w pred')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('NeuralODE: Traiettoria vera vs predetta')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.savefig(folder + 'test' + str(i) + '.png')

# for i in range(0,NTest,25):
#     fig, ax = plt.subplots(1,2,figsize=(15,9))
#     ax[0].plot(tt, testing_target[i,:,0], 'r-', label='v true')
#     ax[0].plot(tt, 5/2*variables3[i,:,0], 'k--', label='v pred')
#     ax[0].plot(tt, testing_target[i,:,1], 'g-', label='w true')
#     ax[0].plot(tt, 5/2*variables3[i,:,1], 'b--', label='w pred')
#     ax[0].set_ylabel('State')
#     ax[0].set_title('NeuralODE: Predicted vs True Trajectory')
#     ax[0].grid(True)
#     ax[0].legend(loc='upper right')

#     ax[1].scatter(variables3[i,:,0],variables3[i,:,1], c=tt, cmap='hot', label='predicted')
#     ax[1].scatter(dataset_testg['out_fields'][i,:,0],dataset_testg['out_fields'][i,:,1], c=tt, cmap='cool', label='true')
#     fig.colorbar(ax[1].collections[0], ax=ax[1], label='Time', fraction=0.046, pad=0.04)
#     fig.colorbar(ax[1].collections[1], ax=ax[1], fraction=0.046, pad=0.04)
#     ind1 = test_triangles[i][0]
#     ind2 = test_triangles[i][1]
#     ind3 = test_triangles[i][2]
#     ax[1].plot([variables3[i,ind1,0],variables3[i,ind2,0]],[variables3[i,ind1,1],variables3[i,ind2,1]], c='red', alpha=0.5)
#     ax[1].plot([variables3[i,ind1,0],variables3[i,ind3,0]],[variables3[i,ind1,1],variables3[i,ind3,1]], c='red', alpha=0.5)
#     ax[1].plot([variables3[i,ind2,0],variables3[i,ind3,0]],[variables3[i,ind2,1],variables3[i,ind3,1]], c='red', alpha=0.5)
#     ax[1].plot([variables3[i,test_indexes[i][0],0],variables3[i,test_indexes[i][1],0]],[variables3[i,test_indexes[i][0],1],variables3[i,test_indexes[i][1],1]], 'r--', label='largest persistence')
#     ax[1].plot([dataset_testg['out_fields'][i,ind1,0],dataset_testg['out_fields'][i,ind2,0]],[dataset_testg['out_fields'][i,ind1,1],dataset_testg['out_fields'][i,ind2,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_testg['out_fields'][i,ind1,0],dataset_testg['out_fields'][i,ind3,0]],[dataset_testg['out_fields'][i,ind1,1],dataset_testg['out_fields'][i,ind3,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_testg['out_fields'][i,ind2,0],dataset_testg['out_fields'][i,ind3,0]],[dataset_testg['out_fields'][i,ind2,1],dataset_testg['out_fields'][i,ind3,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_testg['out_fields'][i,test_indexes[i][0],0],dataset_testg['out_fields'][i,test_indexes[i][1],0]],[dataset_testg['out_fields'][i,test_indexes[i][0],1],dataset_testg['out_fields'][i,test_indexes[i][1],1]], 'b--', label='largest persistence')
#     ax[1].legend()
#     ax[1].set_title('Phase Space')
#     ax[1].axis('equal')
#     plt.savefig(folder + 'test' + str(i) + '.png')

#%%
##########################
# FINE TRAJECTORIES TEST #
##########################
variables4, _, _ = evolve_dynamics_symplectic(dataset_test_fine, x0_test_fine)
for i in range(0,NTest,25):
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,t_max,dt_test), testing_target_fine[i,:,0], 'r-', label='v true')
    ax.plot(np.arange(0,t_max,dt_test), 5/2*variables4[i,:,0], 'k--', label='v pred')
    ax.plot(np.arange(0,t_max,dt_test), testing_target_fine[i,:,1], 'g-', label='w true')
    ax.plot(np.arange(0,t_max,dt_test), 5/2*variables4[i,:,1], 'b--', label='w pred')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('NeuralODE: Traiettoria vera vs predetta')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.savefig(folder + 'test_fine' + str(i) + '.png')

# for i in range(0,NTest,25):
#     fig, ax = plt.subplots(1,2,figsize=(15,9))
#     ax[0].plot(np.arange(0,t_max,dt_test), testing_target_fine[i,:,0], 'r-', label='v true')
#     ax[0].plot(np.arange(0,t_max,dt_test), 5/2*variables4[i,:,0], 'k--', label='v pred')
#     ax[0].plot(np.arange(0,t_max,dt_test), testing_target_fine[i,:,1], 'g-', label='w true')
#     ax[0].plot(np.arange(0,t_max,dt_test), 5/2*variables4[i,:,1], 'b--', label='w pred')
#     ax[0].set_ylabel('State')
#     ax[0].set_title('NeuralODE: Predicted vs True Trajectory')
#     ax[0].grid(True)
#     ax[0].legend(loc='upper right')

#     ax[1].scatter(variables4[i,:,0],variables4[i,:,1], c=np.arange(0,t_max,dt_test), cmap='hot', label='predicted')
#     ax[1].scatter(dataset_test_fine['out_fields'][i,:,0],dataset_test_fine['out_fields'][i,:,1], c=np.arange(0,t_max,dt_test), cmap='cool', label='true')
#     fig.colorbar(ax[1].collections[0], ax=ax[1], label='Time', fraction=0.046, pad=0.04)
#     fig.colorbar(ax[1].collections[1], ax=ax[1], fraction=0.046, pad=0.04)
#     ind1 = test_triangles_fine[i][0]
#     ind2 = test_triangles_fine[i][1]
#     ind3 = test_triangles_fine[i][2]
#     ax[1].plot([variables4[i,ind1,0],variables4[i,ind2,0]],[variables4[i,ind1,1],variables4[i,ind2,1]], c='red', alpha=0.5)
#     ax[1].plot([variables4[i,ind1,0],variables4[i,ind3,0]],[variables4[i,ind1,1],variables4[i,ind3,1]], c='red', alpha=0.5)
#     ax[1].plot([variables4[i,ind2,0],variables4[i,ind3,0]],[variables4[i,ind2,1],variables4[i,ind3,1]], c='red', alpha=0.5)
#     ax[1].plot([variables4[i,test_indexes_fine[i][0],0],variables4[i,test_indexes_fine[i][1],0]],[variables4[i,test_indexes_fine[i][0],1],variables4[i,test_indexes_fine[i][1],1]], 'r--', label='largest persistence')
#     ax[1].plot([dataset_test_fine['out_fields'][i,ind1,0],dataset_test_fine['out_fields'][i,ind2,0]],[dataset_test_fine['out_fields'][i,ind1,1],dataset_test_fine['out_fields'][i,ind2,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_test_fine['out_fields'][i,ind1,0],dataset_test_fine['out_fields'][i,ind3,0]],[dataset_test_fine['out_fields'][i,ind1,1],dataset_test_fine['out_fields'][i,ind3,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_test_fine['out_fields'][i,ind2,0],dataset_test_fine['out_fields'][i,ind3,0]],[dataset_test_fine['out_fields'][i,ind2,1],dataset_test_fine['out_fields'][i,ind3,1]], c='blue', alpha=0.5)
#     ax[1].plot([dataset_test_fine['out_fields'][i,test_indexes_fine[i][0],0],dataset_test_fine['out_fields'][i,test_indexes_fine[i][1],0]],[dataset_test_fine['out_fields'][i,test_indexes_fine[i][0],1],dataset_test_fine['out_fields'][i,test_indexes_fine[i][1],1]], 'b--', label='largest persistence')
#     ax[1].legend()
#     ax[1].set_title('Phase Space')
#     ax[1].axis('equal')

#     plt.savefig(folder + 'test_fine' + str(i) + '.png')


#%%
#########################
# EXTENDED TRAJECTORIES #
#########################
variables_ext, _, _ = evolve_dynamics_symplectic(dataset_test_ext, x0_test_ext)
tt_ext    = t_num_ext[0,:]
tt_comp   = np.setdiff1d(tt_ext, tt)

for i in range(0,NTest_ext,5):
    fig, ax = plt.subplots()
    ax.plot(tt, testing_target_ext[i,:len(tt),0], 'r-', label='v true')
    ax.plot(tt_comp, testing_target_ext[i,len(tt):,0], 'y-', label='v true ext', linewidth=2)
    ax.plot(tt_ext, 5/2*variables_ext[i,:,0], 'k--', label='v pred')
    ax.plot(tt, testing_target_ext[i,:len(tt),1], 'g-', label='w true')
    ax.plot(tt_comp, testing_target_ext[i,len(tt):,1], 'y-', label='w true ext', linewidth=2)
    ax.plot(tt_ext, 5/2*variables_ext[i,:,1], 'b--', label='w pred')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('NeuralODE: Traiettoria vera vs predetta')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.savefig(folder + 'test_extended_' + str(i) + '.png')

#%%
################
# MODEL SAVING #
################
folder_models = 'result_models'

if os.path.exists(folder_models) == False:
    os.mkdir(folder_models)

# NNdyn.save(folder_models + '/MSE_TopoLoss_SympEu.keras')
# model = tf.keras.models.load_model(folder_models + '/MSE_TopoLoss_SympEu.keras')

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
