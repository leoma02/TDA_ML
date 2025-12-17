import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate
import scipy.io
from scipy.integrate import solve_ivp


def normalize_forw(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return (2.0*v - v_min - v_max) / (v_max - v_min)

def normalize_back(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return 0.5*(v_min + v_max + (v_max - v_min) * v)

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max
    
def analyze_normalization(problem, normalization_definition):
    normalization = dict()
    normalization['dt_base'] = normalization_definition['time']['time_constant']
    normalization['x_min'] = np.array(normalization_definition['space']['min'])
    normalization['x_max'] = np.array(normalization_definition['space']['max'])
    if len(problem.get('input_parameters', [])) > 0:
        normalization['inp_parameters_min'] = np.array([normalization_definition['input_parameters'][v['name']]['min'] for v in problem['input_parameters']])
        normalization['inp_parameters_max'] = np.array([normalization_definition['input_parameters'][v['name']]['max'] for v in problem['input_parameters']])
    if len(problem.get('input_signals', [])) > 0:
        normalization['inp_signals_min'] = np.array([normalization_definition['input_signals'][v['name']]['min'] for v in problem['input_signals']])
        normalization['inp_signals_max'] = np.array([normalization_definition['input_signals'][v['name']]['max'] for v in problem['input_signals']])
    normalization['out_fields_min'] = np.array([normalization_definition['output_fields'][v['name']]['min'] for v in problem['output_fields']])
    normalization['out_fields_max'] = np.array([normalization_definition['output_fields'][v['name']]['max'] for v in problem['output_fields']])
    return normalization

def analyze_normalization_epi(problem, normalization_definition):
    normalization = dict()
    normalization['dt_base'] = normalization_definition['time']['time_constant']
    if len(problem.get('input_parameters', [])) > 0:
        normalization['inp_parameters_min'] = np.array([normalization_definition['input_parameters'][v['name']]['min'] for v in problem['input_parameters']])
        normalization['inp_parameters_max'] = np.array([normalization_definition['input_parameters'][v['name']]['max'] for v in problem['input_parameters']])
    if len(problem.get('input_signals', [])) > 0:
        normalization['inp_signals_min'] = np.array([normalization_definition['input_signals'][v['name']]['min'] for v in problem['input_signals']])
        normalization['inp_signals_max'] = np.array([normalization_definition['input_signals'][v['name']]['max'] for v in problem['input_signals']])
    normalization['out_fields_min'] = np.array([normalization_definition['output_fields'][v['name']]['min'] for v in problem['output_fields']])
    normalization['out_fields_max'] = np.array([normalization_definition['output_fields'][v['name']]['max'] for v in problem['output_fields']])
    return normalization

def dataset_normalize(dataset, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    dataset['times']              = dataset['times'] / normalization['dt_base']
    dataset['points']             = normalize_forw(dataset['points']        , normalization['x_min']             , normalization['x_max']             , axis = 1)
    dataset['points_full']        = normalize_forw(dataset['points_full']   , normalization['x_min']             , normalization['x_max']             , axis = 3)
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    dataset['out_fields']         = normalize_forw(dataset['out_fields']    , normalization['out_fields_min']    , normalization['out_fields_max']    , axis = 3)
    
def dataset_normalize_epi(dataset, problem, normalization_definition):
    normalization = analyze_normalization_epi(problem, normalization_definition)
    dataset['times']              = dataset['times'] / normalization['dt_base']
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    if dataset['out_fields'] is not None:
        dataset['out_fields']    = normalize_forw(dataset['out_fields']   , normalization['out_fields_min']   , normalization['out_fields_max']   , axis = 2)
    
def denormalize_output(out_fields, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    return normalize_back(out_fields , normalization['out_fields_min'], normalization['out_fields_max'], axis = 3)
    
def process_dataset_epi(dataset, problem, normalization_definition, dt = None, num_points_subsample = None):
    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
        dataset['times'] = times

    if dataset['inp_signals'] is not None:
        num_samples = dataset['inp_signals'].shape[0]
    else:
        num_samples = dataset['inp_parameters'].shape[0]
    num_times = dataset['times'].shape[0]

    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    dataset_normalize_epi(dataset, problem, normalization_definition)

    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = tf.convert_to_tensor(dataset['inp_parameters'], tf.float64)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals'] = tf.convert_to_tensor(dataset['inp_signals'], tf.float64)
    dataset['beta_state'] = tf.convert_to_tensor(dataset['beta_state'], tf.float64)
    dataset['inf_variables'] = tf.convert_to_tensor(dataset['inf_variables'], tf.float64)
    dataset['target_incidence'] = tf.convert_to_tensor(dataset['target_incidence'], tf.float64)
    dataset['beta_state'] = tf.convert_to_tensor(dataset['beta_state'], tf.float64)
    dataset['time_vec'] = tf.squeeze(tf.convert_to_tensor(dataset['time_vec'], tf.float64))
    dataset['target_cases'] = tf.squeeze(tf.convert_to_tensor(dataset['target_cases'], tf.float64))
    dataset['times'] = tf.squeeze(tf.convert_to_tensor(dataset['times'], tf.float64))
    dataset['initial_state'] = tf.convert_to_tensor(dataset['initial_state'], tf.float64)

def process_dataset_epi_real(dataset, problem, normalization_definition, dt = None, num_points_subsample = None):
    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
        dataset['times'] = times

    if dataset['inp_signals'] is not None:
        num_samples = dataset['inp_signals'].shape[0]
    else:
        num_samples = dataset['inp_parameters'].shape[0]
    num_times = dataset['times'].shape[0]

    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    dataset_normalize_epi(dataset, problem, normalization_definition)

    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = tf.convert_to_tensor(dataset['inp_parameters'], tf.float64)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals'] = tf.convert_to_tensor(dataset['inp_signals'], tf.float64)
        if tf.rank(dataset['inp_signals']) == 1:
            dataset['inp_signals'] = tf.expand_dims(dataset['inp_signals'], axis = 0)
    dataset['time_vec'] = tf.squeeze(tf.convert_to_tensor(dataset['time_vec'], tf.float64))
    #dataset['target'] = tf.squeeze(tf.convert_to_tensor(dataset['target'], tf.float64))
    if tf.rank(dataset['out_fields']) == 1:
        dataset['out_fields'] = tf.expand_dims(dataset['out_fields'], axis = 0)
    dataset['times'] = tf.squeeze(tf.convert_to_tensor(dataset['times'], tf.float64))

def cut_dataset_epi_real(dataset, T_max):
    dataset = dataset.copy()
    n_T_num = int(T_max/(dataset['times'][1] - dataset['times'][0]))
    n_T = int(T_max/(dataset['time_vec'][1] - dataset['time_vec'][0]))
    n_w = int(T_max/7)
    dataset['times'] = dataset['times'][:n_T_num] 
    dataset['inp_signals'] = dataset['inp_signals'][:,:n_T_num,:]
    dataset['target'] = dataset['target'][:, :n_w] 
    dataset['num_times'] = T_max+1
    dataset['time_vec'] = dataset['time_vec'][:n_T]
    dataset['weeks'] = n_w
    return dataset
    
def cut_dataset_epi(dataset, T_max):
    dataset = dataset.copy()
    n_T_num = int(T_max/(dataset['times'][1] - dataset['times'][0]))
    n_T = int(T_max/(dataset['time_vec'][1] - dataset['time_vec'][0]))
    n_w = int(T_max/7)
    dataset['times'] = dataset['times'][:n_T_num] 
    dataset['inp_signals'] = dataset['inp_signals'][:,:n_T_num,:]
    dataset['beta_state'] = dataset['beta_state'][:, :n_T_num,:]
    dataset['inf_variables'] = dataset['inf_variables'][:,:n_T_num,:]
    dataset['target_incidence'] = dataset['target_incidence'][:, :n_w] 
    dataset['target'] = dataset['target'][:, :n_w] 
    dataset['num_times'] = T_max+1
    dataset['time_vec'] = dataset['time_vec'][:n_T]
    dataset['weeks'] = n_w
    return dataset


def traj_gen(omega,T,dt,v0,w0):
    
	def rhs(t, y, omega):
			v, w = y
			dvdt = w
			dwdt = -omega**2 * v
			return [dvdt, dwdt]

	t_span = (0,T)
	t_eval = np.arange(0, T, dt)
	sol    = solve_ivp(rhs,t_span,[v0,w0],args=(omega,),method='BDF',t_eval=t_eval)

	v = sol.y[0]
	w = sol.y[1]

	return v,w


def generate_dataset(NT,normalization,T,dt,seed):
    
    np.random.seed(seed)

    trajectories = []
    x0           = []
    omega_vec    = []

    omega_min = normalization['input_parameters']['omega']['min']
    omega_max = normalization['input_parameters']['omega']['max']

    for i in range(1,NT+1):
        omega = np.random.uniform(omega_min, omega_max)
        v0    = 1.5
        w0    = v0
        x_min = np.array([normalization['output_fields']['v']['min'], normalization['output_fields']['w']['min']])
        x_max = np.array([normalization['output_fields']['v']['max'], normalization['output_fields']['w']['max']])

        v,w = traj_gen(omega,T,dt,v0,w0)
        x_temp = np.hstack((v.reshape(-1,1),w.reshape(-1,1)))
        trajectories.append(x_temp)
        x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

        omega_temp = np.full((len(x_temp), 1), omega, dtype=np.float64)
        omega_vec.append(omega_temp)

    x0     = np.stack(x0, axis=0).astype(np.float64)
    target = np.stack(trajectories, axis=0).astype(np.float64)
    omega  = np.stack(omega_vec, axis=0).astype(np.float64)

    return x0,target,omega
