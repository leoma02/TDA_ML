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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

#%%
#####################
# GLOBAL PLOT STYLE #
#####################
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,

    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "legend.fontsize": 8.5,

    "lines.linewidth": 1.4,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "legend.frameon": False,

    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

#%% 
####################
# MODEL PARAMETERS #
####################
t_max             = 30                                       # time horizon
t_max_ext         = 100                                      # extended time horizon
dt                = 1                                        # real time step 
dt_base           = 1                                        # rescaling factor 
t                 = np.arange(0, t_max+dt, dt)[None,:]       # time values
t_ext             = np.arange(0, t_max_ext+dt, dt)[None,:]   # extended time values
dt_num            = 0.15                                     # numerical time step
dt_test           = dt_num/2                                 # numerical time step for testing
t_num             = np.arange(0, t_max, dt_num)[None, :]     # numerical time values
t_num_fine        = np.arange(0, t_max, dt_test)[None, :]    # numerical time values for fine testing
t_num_ext         = np.arange(0, t_max_ext, dt_num)[None, :] # numerical time values
t_num_ext_fine    = np.arange(0, t_max_ext, dt_test)[None,:] # numerical time values for fine testing
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
############################
# EXPERIMENT CONFIGURATION #
############################
MODEL_NAME = "MSE"

# Available models:
# "MSE"
# "TopoLoss"
# "TopoLossDirect"
# "TopoLossMulti"
# "TopoLossSW"

RUN_MODE = "postprocess"

# Available modes:
# "train"
# "postprocess"

noise_training = True
noise_level    = 0.10
noise_seed     = 1234

lambda_topo    = 1e2
rho_topo       = 0.0225

window_size_sw = 2
delay_sw       = 5
step_sw        = 1

TRANSIENT_TIME = 5.0

RUN_LONG_TIME_ANALYSIS  = True

N_TRAJECTORY_PLOTS = 6
N_LONG_TIME_PLOTS  = 3
N_PHASE_PLOTS      = 3

RUN_PHASE_SPACE_ANALYSIS = True
N_PHASE_PLOTS            = 3

#%%
######################
# GENERATING FOLDERS #
######################

if noise_training:
    SCENARIO_NAME = f"noise_{int(100 * noise_level)}"
else:
    SCENARIO_NAME = "clean"

folder         = os.path.join("results", SCENARIO_NAME, MODEL_NAME)
folder_train   = os.path.join(folder, "train")
folder_figures = os.path.join(folder, "figures")
folder_metrics = os.path.join(folder, "metrics")
folder_models  = os.path.join(folder, "models")

if RUN_MODE == "train":
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder_train, exist_ok=True)
    os.makedirs(folder_figures, exist_ok=True)
    os.makedirs(folder_metrics, exist_ok=True)
    os.makedirs(folder_models, exist_ok=True)

elif RUN_MODE == "postprocess":
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Results folder not found: {folder}"
        )
    os.makedirs(folder_figures, exist_ok=True)
    model_path = os.path.join(folder_models, MODEL_NAME + ".keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")
else:
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

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
NTest_ext = 100 

# Generating training dataset
seed = 0
x0_train, training_target, mi_train = utils.generate_dataset(NTrain, normalization, t_max, dt_num, seed)

# Generating testing dataset
seed = 10
#x0_test, testing_target, mi_test = utils.generate_dataset(NTest, normalization, t_max, dt_num, seed)
x0_test, testing_target, mi_test = utils.generate_test_dataset(NTest, normalization, t_max, dt_num)

# Generating testing dataset
seed = 10
#x0_test_fine, testing_target_fine, mi_test_fine = utils.generate_dataset(NTest, normalization, t_max, dt_test, seed)
x0_test_fine, testing_target_fine, mi_test_fine = utils.generate_test_dataset(NTest, normalization, t_max, dt_test)

# Generating extended testing dataset
seed = 10
x0_test_ext, testing_target_ext, mi_test_ext = utils.generate_test_dataset(NTest_ext, normalization, t_max_ext, dt_num)

# Generating extended testing dataset
seed = 10
x0_test_ext_fine, testing_target_ext_fine, mi_test_ext_fine = utils.generate_test_dataset(NTest_ext, normalization, t_max_ext, dt_test)

#%%
######################################
# ADD NOISE TO TRAINING TRAJECTORIES #
######################################
training_target_clean = None

if noise_training == True:
    # Keep an exact copy for comparisons/plots
    training_target_clean = training_target.copy()

    # Noisy observations used during training
    training_target = utils.add_relative_gaussian_noise(
        training_target_clean,
        noise_level=noise_level,
        seed=noise_seed,
        keep_initial_clean=True
    )

    # Representative training trajectories
    mu_values = mi_train[:, 0, 0]

    target_mu = [
        0.0,
        2.5
    ]

    plot_indices = [
        int(
            np.argmin(
                np.abs(mu_values - target)
            )
        )
        for target in target_mu
    ]
    time_train = np.arange(0, t_max, dt_num)
    fig, ax = plt.subplots(
        len(plot_indices),
        1,
        figsize=(7.0, 1.45 * len(plot_indices) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for row, i in enumerate(plot_indices):
        current_ax = ax[row, 0]
        current_ax.plot(
            time_train,
            training_target_clean[i, :, 0],
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=0.55,
            label=r"$v_{\mathrm{ref}}$"
        )
        current_ax.plot(
            time_train,
            training_target_clean[i, :, 1],
            color="0.45",
            linestyle="--",
            linewidth=1.15,
            alpha=0.55,
            label=r"$w_{\mathrm{ref}}$"
        )
        current_ax.plot(
            time_train,
            training_target[i, :, 0],
            color="C0",
            linestyle="-",
            linewidth=1.00,
            alpha=0.90,
            label=r"$v_{\mathrm{noisy}}$"
        )
        current_ax.plot(
            time_train,
            training_target[i, :, 1],
            color="C1",
            linestyle="-",
            linewidth=1.00,
            alpha=0.90,
            label=r"$w_{\mathrm{noisy}}$"
        )

        # Parameter value
        current_ax.text(
            0.02,
            0.88,
            rf"$\mu={mi_train[i,0,0]:.2f}$",
            transform=current_ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="top"
        )
        current_ax.grid(
            True,
            alpha=0.18,
            linewidth=0.45
        )
        current_ax.tick_params(
            axis="both",
            labelsize=8
        )

    # Common labels
    ax[-1, 0].set_xlabel(
        r"$t$",
        fontsize=10
    )
    fig.supylabel(
        "State",
        fontsize=10
    )

    # Legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=2.4
    )

    # Layout
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.075,
        top=0.94,
        hspace=0.16
    )

    # Save PDF for LaTeX
    fig.savefig(
        os.path.join(
            folder_figures,
            "NoisyTrainingTrajectories.pdf"
        ),
        bbox_inches="tight"
    )

    # Save PNG for quick inspection
    fig.savefig(
        os.path.join(
            folder_figures,
            "NoisyTrainingTrajectories.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

#%%
######################
# DATASET PARAMETERS #
######################
n_size                     = x0_train.shape[0]
n_size_testg               = x0_test.shape[0]
training_var_numpy         = None
testing_var_numpy          = None
testing_var_numpy_fine     = None
testing_var_numpy_ext      = None
inp_params_train           = mi_train[:,0,:]
inp_params_test            = mi_test[:,0,:]
inp_params_test_fine       = mi_test_fine[:,0,:]
inp_params_test_ext        = mi_test_ext[:,0,:]
inp_params_test_ext_fine   = mi_test_ext_fine[:,0,:]

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

dataset_test_ext_fine = {
        'times'         : t_num_ext_fine.T,           # [num_times]
        'inp_parameters': inp_params_test_ext_fine,   # [num_samples x num_par]
        'inp_signals'   : None,                  # [num_samples x num_times x num_signals]
        'out_fields'    : testing_target_ext_fine,    # [num_samples x num_times x num_targets]
        'num_times'     : t_max_ext,
        'time_vec'      : t_ext.T,
        'frac'          : int(dt/dt_test)
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
utils.process_dataset_epi_real(dataset_test_ext_fine, problem, normalization, dt = None, num_points_subsample = None)
print(dataset_train["inp_parameters"].shape)
print(dataset_train["out_fields"].shape)

# Extract topological information from dataset
topological_indexes, topological_triangles = utils.extract_indexes(dataset_train['out_fields'])
topo_tri_new                               = utils.extract_indexes_new(dataset_train['out_fields'], rho_topo)
topo_idx_mat, topo_mask_mat                = utils.build_topological_index_matrix(topo_tri_new)
test_indexes, test_triangles               = utils.extract_indexes(dataset_testg['out_fields'])
test_indexes_fine, test_triangles_fine     = utils.extract_indexes(dataset_test_fine['out_fields'])   
test_tri_new                               = utils.extract_indexes_new(dataset_testg['out_fields'], rho_topo)
test_idx_mat, test_mask_mat                = utils.build_topological_index_matrix(test_tri_new)
test_tri_new_fine                          = utils.extract_indexes_new(dataset_test_fine['out_fields'], rho_topo)
test_idx_mat_fine, test_mask_mat_fine      = utils.build_topological_index_matrix(test_tri_new_fine)

pairs = np.zeros((NTrain, 2), dtype=np.int64)
mask  = np.zeros((NTrain,), dtype=np.float64)
for b, item in enumerate(topological_indexes):
    if len(item) == 2:
        pairs[b, 0] = int(item[0])
        pairs[b, 1] = int(item[1])
        mask[b] = 1.0
    else:
        pairs[b, 0] = 0
        pairs[b, 1] = 0
        mask[b] = 0.0
pairs = tf.constant(pairs, tf.int64)
mask = tf.constant(mask, tf.float64)

# Extract topological information from sw embedded dataset
sw_v_train, sw_w_train = utils.sliding_window(dataset_train['out_fields'], window_size=2, delay=5, step=1)
sw_v_test, sw_w_test   = utils.sliding_window(dataset_testg['out_fields'], window_size=2, delay=5, step=1)
topological_indexes_v, topological_triangles_v = utils.extract_indexes(sw_v_train)
topological_indexes_w, topological_triangles_w = utils.extract_indexes(sw_w_train)
test_indexes_v, test_triangles_v               = utils.extract_indexes(sw_v_test)
test_indexes_w, test_triangles_w               = utils.extract_indexes(sw_w_test)
pairs_v = np.zeros((NTrain, 2), dtype=np.int64)
pairs_w = np.zeros((NTrain, 2), dtype=np.int64)
mask_v  = np.zeros((NTrain,), dtype=np.float64)
mask_w  = np.zeros((NTrain,), dtype=np.float64)

for b, item in enumerate(topological_indexes_v):
    if len(item) == 2:
        pairs_v[b, 0] = int(item[0])
        pairs_v[b, 1] = int(item[1])
        mask_v[b] = 1.0
    else:
        pairs_v[b, 0] = 0
        pairs_v[b, 1] = 0
        mask_v[b] = 0.0
for b, item in enumerate(topological_indexes_w):
    if len(item) == 2:
        pairs_w[b, 0] = int(item[0])
        pairs_w[b, 1] = int(item[1])
        mask_w[b] = 1.0
    else:
        pairs_w[b, 0] = 0
        pairs_w[b, 1] = 0
        mask_w[b] = 0.0
pairs_v = tf.constant(pairs_v, tf.int64)
mask_v = tf.constant(mask_v, tf.float64)
pairs_w = tf.constant(pairs_w, tf.int64)
mask_w = tf.constant(mask_w, tf.float64)

#%%
##############################
# NEURAL OPERATOR DEFINITION #
##############################

if RUN_MODE == "train":
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

    NNdyn = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.tanh, input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(num_latent_states)
    ])
    with open("weights.pkl", "rb") as f:
        good_init = pickle.load(f)
    NNdyn.set_weights(good_init)


elif RUN_MODE == "postprocess":
    model_path = os.path.join(folder_models, MODEL_NAME + ".keras")

    NNdyn = tf.keras.models.load_model(model_path, compile=False)
    print(f"Loaded model: {model_path}")

NNdyn.summary()

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
        lat_state = lat_state + dt_int/dt_ref * NNdyn(tf.concat(inputs, axis = -1))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

def evolve_dynamics_symplectic(dataset, initial_lat_state):
    lat_state         = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size=dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)

    inp_params = dataset['inp_parameters']
    dt_int     = (dataset['times'][1] - dataset['times'][0])

    for i in tf.range(dataset['num_times'] - 1):
        v = lat_state[:, 0:1]
        w = lat_state[:, 1:2]

        f_n   = NNdyn(tf.concat([lat_state, inp_params], axis=-1))
        dw_n  = f_n[:, 1:2]
        w_new = w + dt_int * dw_n

        lat_mixed = tf.concat([v, w_new], axis=-1)
        f_m       = NNdyn(tf.concat([lat_mixed, inp_params], axis=-1))
        dv_m      = f_m[:, 0:1]
        v_new     = v + dt_int * dv_m

        lat_state         = tf.concat([v_new, w_new], axis=-1)
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1, 0, 2))

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

def MSE_topological_loss(dataset, lat_states, pairs, mask):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    MSE   = tf.reduce_mean(tf.square((state) - dataset['out_fields']))

    i  = pairs[:, 0]  # (B,)
    j  = pairs[:, 1]  # (B,)
    si = tf.gather(state, i, axis=1, batch_dims=1)  # (B,2)
    sj = tf.gather(state, j, axis=1, batch_dims=1)  # (B,2)
    ti = tf.gather(dataset['out_fields'],  i, axis=1, batch_dims=1)  # (B,2)
    tj = tf.gather(dataset['out_fields'],  j, axis=1, batch_dims=1)  # (B,2)

    d1 = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2 = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)

    per_traj = tf.square(d1 - d2)                 # (B,)

    masked = per_traj * mask
    denom  = tf.reduce_sum(mask) + 1e-10

    topological_term = tf.reduce_sum(masked) / denom

    return MSE + lambda_topo * topological_term

def MSE_topological_loss_direct(dataset, lat_states, pairs, mask):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    true  = dataset['out_fields']

    MSE = tf.reduce_mean(tf.square(state - true))

    i      = pairs[:, 0]
    j    = pairs[:, 1]
    pred_i = tf.gather(state, i, axis=1, batch_dims=1)
    pred_j = tf.gather(state, j, axis=1, batch_dims=1)
    true_i = tf.gather(true, i, axis=1, batch_dims=1)
    true_j = tf.gather(true, j, axis=1, batch_dims=1)
    dist_i = tf.reduce_sum(tf.square(pred_i - true_i), axis=-1)
    dist_j = tf.reduce_sum(tf.square(pred_j - true_j), axis=-1)

    per_traj = 0.5 * (dist_i + dist_j)

    masked = per_traj * mask
    denom  = tf.reduce_sum(mask) + 1e-10

    topological_term = tf.reduce_sum(masked) / denom

    return MSE + lambda_topo * topological_term

def MSE_topological_loss_multi(dataset, lat_states, topo_idx_mat, topo_mask_mat):
    state = evolve_dynamics_symplectic(dataset, lat_states)
    true  = dataset['out_fields']
    MSE   = tf.reduce_mean(tf.square(state - true))

    state_selected           = tf.gather(state,topo_idx_mat,axis=1,batch_dims=1)
    true_selected            = tf.gather(true,topo_idx_mat,axis=1,batch_dims=1)
    squared_distances        = tf.reduce_sum(tf.square(state_selected - true_selected),axis=-1)
    masked_squared_distances = squared_distances * topo_mask_mat

    denom            = tf.reduce_sum(topo_mask_mat) + 1e-10
    topological_term = tf.reduce_sum(masked_squared_distances) / denom

    return MSE + lambda_topo * topological_term

def MSE_topological_loss_sw(dataset, lat_states, pairs_v, pairs_w, mask_v, mask_w, window_size, delay, step):
    state                = evolve_dynamics_symplectic(dataset, lat_states)
    sw_v, sw_w           = utils.sliding_window(state, window_size, delay, step)
    sw_v_true, sw_w_true = utils.sliding_window(dataset['out_fields'], window_size, delay, step)
    MSE                  = tf.reduce_mean(tf.square(state - dataset['out_fields']))

    i        = pairs_v[:, 0]  # (B,)
    j        = pairs_v[:, 1]  # (B,)
    si       = tf.gather(sw_v, i, axis=1, batch_dims=1)  # (B,2)
    sj       = tf.gather(sw_v, j, axis=1, batch_dims=1)  # (B,2)
    ti       = tf.gather(sw_v_true,  i, axis=1, batch_dims=1)  # (B,2)
    tj       = tf.gather(sw_v_true,  j, axis=1, batch_dims=1)  # (B,2)
    d1       = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2       = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)
    tv       = tf.square(d1 - d2)                 # (B,)
    masked_v = tv * mask_v
    denom_v  = tf.reduce_sum(mask_v) + 1e-10

    i        = pairs_w[:, 0]  # (B,)
    j        = pairs_w[:, 1]  # (B,)
    si       = tf.gather(sw_w, i, axis=1, batch_dims=1)  # (B,2)
    sj       = tf.gather(sw_w, j, axis=1, batch_dims=1)  # (B,2)
    ti       = tf.gather(sw_w_true,  i, axis=1, batch_dims=1)  # (B,2)
    tj       = tf.gather(sw_w_true,  j, axis=1, batch_dims=1)  # (B,2)
    d1       = tf.sqrt(tf.reduce_sum(tf.square(si - sj), axis=-1))  # (B,)
    d2       = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1))  # (B,)
    tw       = tf.square(d1 - d2)                 # (B,)
    masked_w = tw * mask_w
    denom_w  = tf.reduce_sum(mask_w) + 1e-10

    return MSE + lambda_topo*(tf.reduce_sum(masked_v) / denom_v + tf.reduce_sum(masked_w) / denom_w)

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

#%%
################
# LOSS WEIGHTS #
################

nu_loss_train = 3e-2 # weight MSE metric
alpha_reg     = 1e-8 # regularization of trainable variables

#%%
######################
# TRAINING FUNCTIONS #
######################

trainable_variables_train = NNdyn.variables

def loss_train():
    l = nu_loss_train * loss_MSE(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_train_symplectic():
    l = nu_loss_train * loss_MSE_symplectic(dataset_train, x0_train) + alpha_reg * weights_reg(NNdyn)
    return l

def loss_train_topological():
    if MODEL_NAME == "MSE":
        objective = loss_MSE_symplectic(dataset_train, x0_train)
    elif MODEL_NAME == "TopoLoss":
        objective = MSE_topological_loss(dataset_train, x0_train, pairs, mask)
    elif MODEL_NAME == "TopoLossDirect":
        objective = MSE_topological_loss_direct(dataset_train, x0_train, pairs, mask)
    elif MODEL_NAME == "TopoLossMulti":
        objective = MSE_topological_loss_multi(dataset_train, x0_train, topo_idx_mat, topo_mask_mat)
    elif MODEL_NAME == "TopoLossSW":
        objective = MSE_topological_loss_sw(dataset_train, x0_train, pairs_v, pairs_w, mask_v, mask_w, window_size_sw, delay_sw, step_sw)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

    return (nu_loss_train * objective + alpha_reg * weights_reg(NNdyn))

def loss_valid():
    # l2 = MSE_topological_loss_multi(dataset_train, x0_train, topo_idx_mat, topo_mask_mat)
    return 0,0

def val_train():
    l = loss_MSE(dataset_train, x0_train)
    return l 

def loss_valid_ext():
    l = loss_MSE(dataset_test_ext, x0_test_ext)
    return l

val_metric = loss_valid

#%%
############
# TRAINING #
############

if RUN_MODE == "train":

    losses_dict = {'Standard': loss_train_symplectic, 'TopoLoss': loss_train_topological}
    opt_train   = optimization.OptimizationProblem(trainable_variables_train, losses_dict, val_metric)

    num_epochs_Adam_train      = 2500
    num_epochs_BFGS_train      = 3000
    num_epochs_BFGS_train_topo = 5000

    print('training (Adam)...')
    opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-2))

    print('training (Adam)...')
    opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-3))

    print('training (Adam)...')
    opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-3))

    print('training (BFGS)...')
    opt_train.optimize_BFGS(num_epochs_BFGS_train)

    opt_train.set_loss_train('TopoLoss')

    print('training (BFGS - final objective)...')
    opt_train.optimize_BFGS(num_epochs_BFGS_train_topo)

elif RUN_MODE == "postprocess":
    print("\n========================================")
    print("POSTPROCESSING MODE")
    print("Training skipped.")
    print("========================================\n")

#%%
#####################
# TEST TRAJECTORIES #
#####################
variables3 = evolve_dynamics_symplectic(dataset_testg, x0_test)
variables4 = evolve_dynamics_symplectic(dataset_test_fine, x0_test_fine)

# Training time step
metrics_base = utils.compute_test_metrics(
    predictions=variables3,
    dataset=dataset_testg,
    mi_values=mi_test[:, 0, 0],
    problem=problem,
    normalization_definition=normalization,
    transient_time=TRANSIENT_TIME
)
metrics_base["time_step"] = dt_num

# Testing time step
metrics_fine = utils.compute_test_metrics(
    predictions=variables4,
    dataset=dataset_test_fine,
    mi_values=mi_test_fine[:, 0, 0],
    problem=problem,
    normalization_definition=normalization,
    transient_time=TRANSIENT_TIME
)
metrics_fine["time_step"] = dt_test

# Summary Statistics
summary_base = utils.metric_summary(metrics_base, r"$\Delta t$")
summary_fine = utils.metric_summary(metrics_fine, r"$\Delta t/2$")
summary      = pd.concat([summary_base,summary_fine],ignore_index=True)
summary.to_csv(os.path.join(folder_metrics,"metrics_summary.csv"),index=False)

# Save results
if RUN_MODE == "train":
    metrics_base.to_csv(os.path.join(folder_metrics, "metrics_dt.csv"), index=False)
    metrics_fine.to_csv(os.path.join(folder_metrics, "metrics_dt_half.csv"), index=False)
    summary.to_csv(os.path.join(folder_metrics, "metrics_summary.csv"), index=False)

# Print
print("\n========================================")
print("TEST METRICS SUMMARY")
print("========================================")
print(summary.to_string(index=False))

#%%
######################
# PLOTTING FUNCTIONS #
######################

def plot_trajectory_comparison(prediction_base, dataset_base, mi_base, prediction_fine, dataset_fine, mi_fine, n_plots, filename, linewidth=1.25):
    plot_indexes = np.linspace(0, len(mi_base) - 1, n_plots, dtype=int)
    time_base    = utils.get_time_axis(dataset_base)
    time_fine    = utils.get_time_axis(dataset_fine)
    true_base    = utils.denormalize_output_epi(dataset_base["out_fields"], problem, normalization)
    pred_base    = utils.denormalize_output_epi(prediction_base, problem, normalization)
    true_fine    = utils.denormalize_output_epi(dataset_fine["out_fields"], problem, normalization)
    pred_fine    = utils.denormalize_output_epi(prediction_fine, problem, normalization)

    # Axis limits: computed independently for the two columns
    selected_values_base = []
    selected_values_fine = []

    for idx in plot_indexes:
        selected_values_base.extend([
            true_base[idx].ravel(),
            pred_base[idx].ravel()
        ])
        selected_values_fine.extend([
            true_fine[idx].ravel(),
            pred_fine[idx].ravel()
        ])

    selected_values_base = np.concatenate(
        selected_values_base
    )
    selected_values_fine = np.concatenate(
        selected_values_fine
    )

    # Base time-step limits
    y_min_base = np.min(
        selected_values_base
    )
    y_max_base = np.max(
        selected_values_base
    )
    y_pad_base = 0.05 * (
        y_max_base - y_min_base
    )
    y_min_base -= y_pad_base
    y_max_base += y_pad_base


    # Fine time-step limits
    y_min_fine = np.min(
        selected_values_fine
    )

    y_max_fine = np.max(
        selected_values_fine
    )

    y_pad_fine = 0.05 * (
        y_max_fine - y_min_fine
    )

    y_min_fine -= y_pad_fine
    y_max_fine += y_pad_fine

    # Figure
    fig_height = 1.45 * n_plots + 0.6

    fig, ax = plt.subplots(
        n_plots,
        2,
        figsize=(7.0, fig_height),
        sharex=True,
        sharey=False,
        squeeze=False
    )

    for row, idx in enumerate(plot_indexes):

        # Base time step
        ax[row, 0].plot(
            time_base,
            true_base[idx, :, 0],
            color="black",
            linestyle="-",
            linewidth=linewidth,
            label=r"$v_{\mathrm{ref}}$"
        )
        ax[row, 0].plot(
            time_base,
            pred_base[idx, :, 0],
            color="C0",
            linestyle="--",
            linewidth=linewidth,
            label=r"$v_{\mathrm{pred}}$"
        )
        ax[row, 0].plot(
            time_base,
            true_base[idx, :, 1],
            color="0.50",
            linestyle="-",
            linewidth=linewidth,
            label=r"$w_{\mathrm{ref}}$"
        )
        ax[row, 0].plot(
            time_base,
            pred_base[idx, :, 1],
            color="C1",
            linestyle="--",
            linewidth=linewidth,
            label=r"$w_{\mathrm{pred}}$"
        )

        # Fine time step
        ax[row, 1].plot(
            time_fine,
            true_fine[idx, :, 0],
            color="black",
            linestyle="-",
            linewidth=linewidth
        )
        ax[row, 1].plot(
            time_fine,
            pred_fine[idx, :, 0],
            color="C0",
            linestyle="--",
            linewidth=linewidth
        )
        ax[row, 1].plot(
            time_fine,
            true_fine[idx, :, 1],
            color="0.50",
            linestyle="-",
            linewidth=linewidth
        )
        ax[row, 1].plot(
            time_fine,
            pred_fine[idx, :, 1],
            color="C1",
            linestyle="--",
            linewidth=linewidth
        )

        # Row settings
        ax[row, 0].set_ylim(
            y_min_base,
            y_max_base
        )
        ax[row, 1].set_ylim(
            y_min_fine,
            y_max_fine
        )

        for col in range(2):
            ax[row, col].grid(
                True,
                alpha=0.18,
                linewidth=0.45
            )
            ax[row, col].tick_params(
                axis="both",
                labelsize=8
            )

        # Parameter value shown once per row
        ax[row, 0].text(
            0.02,
            0.88,
            rf"$\mu={mi_base[idx,0,0]:.2f}$",
            transform=ax[row, 0].transAxes,
            fontsize=8.5,
            ha="left",
            va="top"
        )

    # Time steps
    dt_base_plot = time_base[1] - time_base[0]
    dt_fine_plot = time_fine[1] - time_fine[0]

    # Bottom labels
    ax[-1, 0].set_xlabel(rf"$t$   ($\Delta t = {dt_base_plot:.3f}$)", fontsize=10)
    ax[-1, 1].set_xlabel(rf"$t$   ($\Delta t = {dt_fine_plot:.3f}$)", fontsize=10)
    fig.supylabel("State", fontsize=10)

    # Legend
    handles, labels = ax[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=8.5
    )

    # Layout
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.075,
        top=0.94,
        hspace=0.18,
        wspace=0.08
    )

    # Save
    fig.savefig(
        os.path.join(folder_figures, filename + ".pdf"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder_figures, filename + ".png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def get_first_multi_pair(index_matrix, mask_matrix, trajectory_index):

    indexes = index_matrix[trajectory_index]
    mask    = mask_matrix[trajectory_index]

    if hasattr(indexes, "numpy"):
        indexes = indexes.numpy()

    if hasattr(mask, "numpy"):
        mask = mask.numpy()

    indexes = np.asarray(indexes).reshape(-1)
    mask    = np.asarray(mask).reshape(-1)

    valid_indexes = indexes[mask > 0.5]

    # Remove possible repeated indexes while preserving the original order
    unique_indexes = []

    for index in valid_indexes:

        index = int(index)

        if index not in unique_indexes:
            unique_indexes.append(index)

    if len(unique_indexes) < 2:
        return None

    return [
        unique_indexes[0],
        unique_indexes[1]
    ]

def get_phase_space_pair(
    model_name,
    trajectory_index,
    pairwise_indexes,
    multi_index_matrix=None,
    multi_mask_matrix=None
):

    # MSE, TopoLoss and TopoLossDirect:
    # use the representative pair selected from the reference trajectory
    if model_name in [
        "MSE",
        "TopoLoss",
        "TopoLossDirect"
    ]:

        pair = pairwise_indexes[
            trajectory_index
        ]

        if pair is None or len(pair) != 2:
            return None

        return [
            int(pair[0]),
            int(pair[1])
        ]


    # TopoLossMulti:
    # visualize the segment joining the first two valid selected indexes
    elif model_name == "TopoLossMulti":

        return get_first_multi_pair(
            multi_index_matrix,
            multi_mask_matrix,
            trajectory_index
        )


    # TopoLossSW:
    # its topological constraint is defined in sliding-window space,
    # therefore no phase-space pair corresponds directly to the loss
    elif model_name == "TopoLossSW":

        return None


    else:

        raise ValueError(
            f"Unknown MODEL_NAME: {model_name}"
        )

def plot_selected_segment(
    ax,
    reference_trajectory,
    predicted_trajectory,
    indexes
):

    if indexes is None or len(indexes) != 2:
        return

    ind1 = indexes[0]
    ind2 = indexes[1]

    # Reference segment
    ax.plot(
        [
            reference_trajectory[ind1, 0],
            reference_trajectory[ind2, 0]
        ],
        [
            reference_trajectory[ind1, 1],
            reference_trajectory[ind2, 1]
        ],
        color="0.25",
        linestyle="-",
        linewidth=2.80,
        zorder=6
    )

    ax.scatter(
        [
            reference_trajectory[ind1, 0],
            reference_trajectory[ind2, 0]
        ],
        [
            reference_trajectory[ind1, 1],
            reference_trajectory[ind2, 1]
        ],
        color="0.25",
        marker="o",
        s=28,
        edgecolor="white",
        linewidth=0.5,
        zorder=7
    )


    # Predicted segment evaluated at the SAME indexes
    ax.plot(
        [
            predicted_trajectory[ind1, 0],
            predicted_trajectory[ind2, 0]
        ],
        [
            predicted_trajectory[ind1, 1],
            predicted_trajectory[ind2, 1]
        ],
        color="C1",
        linestyle="--",
        linewidth=2.80,
        zorder=6
    )

    ax.scatter(
        [
            predicted_trajectory[ind1, 0],
            predicted_trajectory[ind2, 0]
        ],
        [
            predicted_trajectory[ind1, 1],
            predicted_trajectory[ind2, 1]
        ],
        color="C1",
        marker="s",
        s=28,
        edgecolor="white",
        linewidth=0.5,
        zorder=7
    )

def plot_phase_space_comparison(
    prediction_base,
    dataset_base,
    mi_base,
    prediction_fine,
    dataset_fine,
    mi_fine,
    reference_indexes_base,
    reference_indexes_fine,
    multi_indexes_base=None,
    multi_mask_base=None,
    multi_indexes_fine=None,
    multi_mask_fine=None,
    n_plots=3,
    filename="PhaseSpaceComparison"
):

    plot_indexes = np.linspace(
        0,
        len(mi_base) - 1,
        n_plots,
        dtype=int
    )

    true_base = utils.denormalize_output_epi(
        dataset_base["out_fields"],
        problem,
        normalization
    )

    pred_base = utils.denormalize_output_epi(
        prediction_base,
        problem,
        normalization
    )

    true_fine = utils.denormalize_output_epi(
        dataset_fine["out_fields"],
        problem,
        normalization
    )

    pred_fine = utils.denormalize_output_epi(
        prediction_fine,
        problem,
        normalization
    )

    # Axis limits
    selected_values = []

    for idx in plot_indexes:

        selected_values.extend([
            true_base[idx],
            pred_base[idx],
            true_fine[idx],
            pred_fine[idx]
        ])

    selected_values = np.concatenate(
        selected_values,
        axis=0
    )

    v_min_plot = np.min(
        selected_values[:, 0]
    )

    v_max_plot = np.max(
        selected_values[:, 0]
    )

    w_min_plot = np.min(
        selected_values[:, 1]
    )

    w_max_plot = np.max(
        selected_values[:, 1]
    )

    v_pad = 0.05 * (
        v_max_plot - v_min_plot
    )

    w_pad = 0.05 * (
        w_max_plot - w_min_plot
    )

    v_min_plot -= v_pad
    v_max_plot += v_pad

    w_min_plot -= w_pad
    w_max_plot += w_pad


    # Figure
    fig, ax = plt.subplots(
        n_plots,
        2,
        figsize=(7.0, 2.65 * n_plots),
        squeeze=False
    )


    for row, idx in enumerate(plot_indexes):

        # Reference-selected indexes
        pair_base = get_phase_space_pair(
            model_name=MODEL_NAME,
            trajectory_index=idx,
            pairwise_indexes=reference_indexes_base,
            multi_index_matrix=multi_indexes_base,
            multi_mask_matrix=multi_mask_base
        )

        pair_fine = get_phase_space_pair(
            model_name=MODEL_NAME,
            trajectory_index=idx,
            pairwise_indexes=reference_indexes_fine,
            multi_index_matrix=multi_indexes_fine,
            multi_mask_matrix=multi_mask_fine
        )


        # Base time step
        ax[row, 0].plot(
            true_base[idx, :, 0],
            true_base[idx, :, 1],
            color="black",
            linestyle="-",
            linewidth=1.25,
            label="Reference trajectory"
        )

        ax[row, 0].plot(
            pred_base[idx, :, 0],
            pred_base[idx, :, 1],
            color="C0",
            linestyle="--",
            linewidth=1.25,
            label="Predicted trajectory"
        )

        plot_selected_segment(
            ax[row, 0],
            true_base[idx],
            pred_base[idx],
            pair_base
        )


        # Fine time step
        ax[row, 1].plot(
            true_fine[idx, :, 0],
            true_fine[idx, :, 1],
            color="black",
            linestyle="-",
            linewidth=1.25
        )

        ax[row, 1].plot(
            pred_fine[idx, :, 0],
            pred_fine[idx, :, 1],
            color="C0",
            linestyle="--",
            linewidth=1.25
        )

        plot_selected_segment(
            ax[row, 1],
            true_fine[idx],
            pred_fine[idx],
            pair_fine
        )


        # Row settings
        for col in range(2):

            ax[row, col].set_xlim(
                v_min_plot,
                v_max_plot
            )

            ax[row, col].set_ylim(
                w_min_plot,
                w_max_plot
            )

            ax[row, col].set_aspect(
                "auto"
            )

            ax[row, col].set_box_aspect(
                0.80
            )

            ax[row, col].grid(
                True,
                alpha=0.18,
                linewidth=0.45
            )

            ax[row, col].tick_params(
                axis="both",
                labelsize=8
            )


        # Parameter value shown once per row
        ax[row, 0].text(
            0.03,
            0.92,
            rf"$\mu={mi_base[idx,0,0]:.2f}$",
            transform=ax[row, 0].transAxes,
            fontsize=8.5,
            ha="left",
            va="top"
        )


    # Time steps
    time_base = utils.get_time_axis(
        dataset_base
    )

    time_fine = utils.get_time_axis(
        dataset_fine
    )

    dt_base_plot = (
        time_base[1]
        -
        time_base[0]
    )

    dt_fine_plot = (
        time_fine[1]
        -
        time_fine[0]
    )


    # Bottom labels
    ax[-1, 0].set_xlabel(
        rf"$v$   ($\Delta t = {dt_base_plot:.3f}$)",
        fontsize=10
    )

    ax[-1, 1].set_xlabel(
        rf"$v$   ($\Delta t = {dt_fine_plot:.3f}$)",
        fontsize=10
    )

    fig.supylabel(
        r"$w$",
        fontsize=10
    )


    # Legend
    legend_handles = [

        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Reference trajectory"
        ),

        Line2D(
            [0], [0],
            color="C0",
            linestyle="--",
            linewidth=1.5,
            label="Predicted trajectory"
        ),

        Line2D(
            [0], [0],
            color="0.25",
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2.8,
            label="Reference selected segment"
        ),

        Line2D(
            [0], [0],
            color="C1",
            linestyle="--",
            marker="s",
            markersize=5,
            linewidth=2.8,
            label="Predicted corresponding segment"
        )
    ]


    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        columnspacing=1.5,
        handlelength=2.6
    )


    # Layout
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.09,
        top=0.90,
        hspace=0.30,
        wspace=0.12
    )


    # Save
    fig.savefig(
        os.path.join(
            folder_figures,
            filename + ".pdf"
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            folder_figures,
            filename + ".png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

#%%
#############################################
# PLOT CREATION FOR TRAJECTORIES COMPARISON #
#############################################
plot_trajectory_comparison(
    prediction_base=variables3,
    dataset_base=dataset_testg,
    mi_base=mi_test,
    prediction_fine=variables4,
    dataset_fine=dataset_test_fine,
    mi_fine=mi_test_fine,
    n_plots=N_TRAJECTORY_PLOTS,
    filename="TrajectoryComparison",
    linewidth=1.25
)

#%%
#%%
########################
# PHASE SPACE ANALYSIS #
########################
if RUN_PHASE_SPACE_ANALYSIS:
    if MODEL_NAME == "TopoLossSW":
        print(
            "Phase-space selected-segment plot skipped for TopoLossSW: "
            "its topology-aware constraint is defined in the "
            "sliding-window embedding space."
        )

    else:
        plot_phase_space_comparison(
            prediction_base=variables3,
            dataset_base=dataset_testg,
            mi_base=mi_test,

            prediction_fine=variables4,
            dataset_fine=dataset_test_fine,
            mi_fine=mi_test_fine,

            reference_indexes_base=test_indexes,
            reference_indexes_fine=test_indexes_fine,

            multi_indexes_base=test_idx_mat,
            multi_mask_base=test_mask_mat,

            multi_indexes_fine=test_idx_mat_fine,
            multi_mask_fine=test_mask_mat_fine,

            n_plots=N_PHASE_PLOTS,
            filename="PhaseSpaceComparison"
        )

#%%
#########################
# EXTENDED TRAJECTORIES #
#########################
variables_ext      = evolve_dynamics_symplectic(dataset_test_ext, x0_test_ext)
variables_ext_fine = evolve_dynamics_symplectic(dataset_test_ext_fine, x0_test_ext_fine)

if RUN_LONG_TIME_ANALYSIS:
    plot_trajectory_comparison(
        prediction_base=variables_ext,
        dataset_base=dataset_test_ext,
        mi_base=mi_test_ext,
        prediction_fine=variables_ext_fine,
        dataset_fine=dataset_test_ext_fine,
        mi_fine=mi_test_ext_fine,
        n_plots=N_LONG_TIME_PLOTS,
        filename="LongTimeTrajectoryComparison",
        linewidth=0.90
    )

if RUN_LONG_TIME_ANALYSIS:
    metrics_ext = utils.compute_test_metrics(
        predictions=variables_ext,
        dataset=dataset_test_ext,
        mi_values=mi_test_ext[:, 0, 0],
        problem=problem,
        normalization_definition=normalization,
        transient_time=TRANSIENT_TIME
    )
    metrics_ext["time_step"] = dt_num

    metrics_ext_fine = utils.compute_test_metrics(
        predictions=variables_ext_fine,
        dataset=dataset_test_ext_fine,
        mi_values=mi_test_ext_fine[:, 0, 0],
        problem=problem,
        normalization_definition=normalization,
        transient_time=TRANSIENT_TIME
    )
    metrics_ext_fine["time_step"] = dt_test

    if RUN_MODE == "train":
        metrics_ext.to_csv(os.path.join(folder_metrics, "metrics_long_dt.csv"), index=False)
        metrics_ext_fine.to_csv(os.path.join(folder_metrics, "metrics_long_dt_half.csv"), index=False)


#%%
################
# MODEL SAVING #
################
if RUN_MODE == "train":
    model_path = os.path.join(folder_models, MODEL_NAME + ".keras")
    NNdyn.save(model_path)
    print(f"Final model saved in: {model_path}")

# %%
