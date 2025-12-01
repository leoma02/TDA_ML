import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate
import scipy.io

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

    num_samples = dataset['inp_signals'].shape[0]
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

    num_samples = dataset['inp_signals'].shape[0]
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

def traj_computation(epsilon,a,b,Iext,theta,T,dt,v0,w0,plot=1):
  def F(x):
    v,w = x
    dv  = v - (v**3)/3 - w + Iext
    dw  = epsilon*(v + a - b*w)
    return np.array([dv,dw])

  def J(x):
    v,w = x
    J11 = 1 - theta*dt*(1-v**2)
    J12 = dt*theta
    J21 = -dt*theta*epsilon
    J22 = 1 + dt*theta*b*epsilon
    return np.array([[J11,J12],[J21,J22]])

  def Newton(x,f,maxit,tol):
    i = 0
    converged = False
    x_old = x.copy()

    def g(x_current):
      return x_current - f(x_current)

    while not converged:
      i += 1
      Jacobian_g = J(x_old)
      delta_x = np.linalg.solve(Jacobian_g, -g(x_old))
      x_new = x_old + delta_x
      if i >= maxit or np.linalg.norm(g(x_new)) < tol or np.linalg.norm(delta_x) < tol:
          converged = True
      else:
          x_old = x_new.copy()

    return x_new

  tt    = np.arange(0,T,dt)
  x     = np.array([v0,w0])

  x_old = x.copy()
  for t in tt[1:]:
    C = x_old + dt*(1-theta)*F(x_old)
    def RHS(x):
      return C + dt*theta*F(x)
    x_new = Newton(x_old,RHS,10,1e-7)
    x = np.vstack((x,x_new))
    x_old = x_new.copy()

  if plot == 1:
    plt.figure(figsize=(10, 6))
    plt.plot(tt, x[:, 0], label='v')
    plt.plot(tt, x[:, 1], label='w')
    plt.xlabel('Time (tt)')
    plt.ylabel('Values (v, w)')
    plt.title('Plot of v and w against Time')
    plt.legend()
    plt.grid(True)
    plt.show()

  return x


def generate_dataset(NT,normalization,theta,T,dt,seed):
    
    np.random.seed(seed)

    trajectories = []
    inputs       = []
    x0           = []
    a_vec        = []
    b_vec        = []
    eps_vec      = []

    eps_min = normalization['input_parameters']['epsilon']['min']
    eps_max = normalization['input_parameters']['epsilon']['max']
    a_min   = normalization['input_parameters']['a']['min']
    a_max   = normalization['input_parameters']['a']['max']
    b_min   = normalization['input_parameters']['b']['min']
    b_max   = normalization['input_parameters']['b']['max']
    u_min   = normalization['input_signals']['I_ext']['min']
    u_max   = normalization['input_signals']['I_ext']['max']

    for i in range(1,NT+1):
        epsilon = np.random.uniform(eps_min, eps_max)
        a       = np.random.uniform(a_min, a_max)
        b       = np.random.uniform(b_min, b_max)
        Iext    = np.random.uniform(u_min, u_max)
        v0      = 1.5
        w0      = v0
        x_min = np.array([normalization['output_fields']['v']['min'], normalization['output_fields']['w']['min']])
        x_max = np.array([normalization['output_fields']['v']['max'], normalization['output_fields']['w']['max']])

        x_temp = traj_computation(epsilon,a,b,Iext,theta,T,dt,v0,w0,0)
        trajectories.append(x_temp)
        x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

        u        = np.full((len(x_temp), 1), (2.0*Iext-u_min-u_max)/(u_max-u_min), dtype=np.float64)
        a_temp   = np.full((len(x_temp), 1), (2.0*a-a_min-a_max)/(a_max-a_min), dtype=np.float64)
        b_temp   = np.full((len(x_temp), 1), (2.0*b-b_min-b_max)/(b_max-b_min), dtype=np.float64)
        eps_temp = np.full((len(x_temp), 1), (2.0*epsilon-eps_min-eps_max)/(eps_max-eps_min), dtype=np.float64)
        a_vec.append(a_temp)
        b_vec.append(b_temp)
        eps_vec.append(eps_temp)
        inputs.append(u)

    x0     = np.stack(x0, axis=0).astype(np.float64)
    u      = np.stack(inputs, axis=0).astype(np.float64)
    target = np.stack(trajectories, axis=0).astype(np.float64)
    a      = np.stack(a_vec, axis=0).astype(np.float64)
    b      = np.stack(b_vec, axis=0).astype(np.float64)
    eps    = np.stack(eps_vec, axis=0).astype(np.float64)

    return x0, u, target, a, b, eps

def check_limit_cycle(trajectory, percentage):
	check_v = np.min(trajectory[:,0]) + percentage*(np.max(trajectory[:,0]) - np.min(trajectory[:,0]))
	check_w = np.min(trajectory[:,1]) + percentage*(np.max(trajectory[:,1]) - np.min(trajectory[:,1]))

	bool_v = trajectory[0,0] > check_v
	bool_w = trajectory[0,1] > check_w
	num_v   = 0
	num_w   = 0

	for i in range(trajectory.shape[0]):
		if (trajectory[i,0] > check_v) != bool_v:
			bool_v = not bool_v
			num_v += 1
		if (trajectory[i,1] > check_w) != bool_w:
			bool_w = not bool_w
			num_w += 1
  
	if num_v > 3 and num_w > 3:
		return True
	else:
		return False
