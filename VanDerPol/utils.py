import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate
from scipy.integrate import solve_ivp
import gudhi as gd
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


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


def traj_gen(mi,t0,T,dt,v0,w0):
    
	def rhs(t, y, mi):
			v, w = y
			dvdt = w
			dwdt = mi*(1-v**2)*w - v
			return [dvdt, dwdt]

	t_span = (t0,T)
	t_eval = np.arange(t0, T, dt)
	sol    = solve_ivp(rhs,t_span,[v0,w0],args=(mi,),method='BDF',t_eval=t_eval)

	v = sol.y[0]
	w = sol.y[1]

	return v,w


def generate_dataset(NT,normalization,T,dt,seed):
    
    np.random.seed(seed)

    trajectories = []
    x0           = []
    mi_vec       = []

    mi_min = normalization['input_parameters']['mi']['min']
    mi_max = normalization['input_parameters']['mi']['max']

    for i in range(1,NT+1):
        mi = np.random.uniform(mi_min, mi_max)
        v0    = 1.5
        w0    = v0
        x_min = np.array([normalization['output_fields']['v']['min'], normalization['output_fields']['w']['min']])
        x_max = np.array([normalization['output_fields']['v']['max'], normalization['output_fields']['w']['max']])

        v,w = traj_gen(mi,0,T,dt,v0,w0)
        x_temp = np.hstack((v.reshape(-1,1),w.reshape(-1,1)))
        trajectories.append(x_temp)
        x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

        mi_temp = np.full((len(x_temp), 1), mi, dtype=np.float64)
        mi_vec.append(mi_temp)

    x0     = np.stack(x0, axis=0).astype(np.float64)
    target = np.stack(trajectories, axis=0).astype(np.float64)
    mi  = np.stack(mi_vec, axis=0).astype(np.float64)

    return x0,target,mi

def generate_shift_dataset(NT,normalization,T,dt,seed):
    
    np.random.seed(seed)

    trajectories = []
    x0           = []
    mi_vec       = []
    t0_vec       = []

    mi_min = normalization['input_parameters']['mi']['min']
    mi_max = normalization['input_parameters']['mi']['max']

    rng = np.random.default_rng()

    for i in range(1,NT+1):
        mi    = np.random.uniform(mi_min, mi_max)
        v0    = 1.5
        w0    = v0
        x_min = np.array([normalization['output_fields']['v']['min'], normalization['output_fields']['w']['min']])
        x_max = np.array([normalization['output_fields']['v']['max'], normalization['output_fields']['w']['max']])

        offset    = rng.integers(3)
        if offset != 0:
            v_temp, w_temp = traj_gen(mi,0,dt*offset/2,dt/10,v0,w0)
            v0_true = v_temp[-1]
            w0_true = w_temp[-1]
        else:
            v0_true = v0
            w0_true = w0
        v,w = traj_gen(mi,dt*offset/2,T+dt*offset/2,dt,v0_true,w0_true)
        x_temp = np.hstack((v.reshape(-1,1),w.reshape(-1,1)))
        trajectories.append(x_temp)
        x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

        mi_temp = np.full((len(x_temp), 1), mi, dtype=np.float64)
        mi_vec.append(mi_temp)
        t0_vec.append(dt*offset/2)

    x0     = np.stack(x0, axis=0).astype(np.float64)
    target = np.stack(trajectories, axis=0).astype(np.float64)
    mi  = np.stack(mi_vec, axis=0).astype(np.float64)
    t0  = np.stack(t0_vec, axis=0).astype(np.float64)
    return x0,target,mi,t0

def generate_test_dataset(NT,normalization,T,dt):

    trajectories = []
    x0           = []
    mi_vec       = []

    mi_min = normalization['input_parameters']['mi']['min']
    mi_max = normalization['input_parameters']['mi']['max']

    for i in range(1,NT+1):
        mi    = mi_max - (mi_max - mi_min) * (i-1) / (NT-1)
        v0    = 1.5
        w0    = v0
        x_min = np.array([normalization['output_fields']['v']['min'], normalization['output_fields']['w']['min']])
        x_max = np.array([normalization['output_fields']['v']['max'], normalization['output_fields']['w']['max']])

        v,w = traj_gen(mi,0,T,dt,v0,w0)
        x_temp = np.hstack((v.reshape(-1,1),w.reshape(-1,1)))
        trajectories.append(x_temp)
        x0.append( (2.0*x_temp[0]-x_min-x_max)/(x_max-x_min) )

        mi_temp = np.full((len(x_temp), 1), mi, dtype=np.float64)
        mi_vec.append(mi_temp)

    x0     = np.stack(x0, axis=0).astype(np.float64)
    target = np.stack(trajectories, axis=0).astype(np.float64)
    mi  = np.stack(mi_vec, axis=0).astype(np.float64)

    return x0,target,mi

def extract_indexes(tj):

    indexes   = []
    triangles = []
    for i in range(tj.shape[0]):
        aC      = gd.AlphaComplex(points=tj[i,:,:])
        st_target = aC.create_simplex_tree()
        st_target.compute_persistence()
        max_per = 0
        max_ind = ()
        for birth_s, death_s in st_target.persistence_pairs():
            if birth_s is not None and death_s is not None and len(birth_s) == 2 and len(death_s) == 3:
                b = float(st_target.filtration(birth_s))
                d = float(st_target.filtration(death_s))
                if d-b > max_per:
                    max_per = d-b
                    max_ind = death_s

        if len(max_ind) == 0:
            indexes.append(())
            triangles.append(())
        else:
            ind1 = max_ind[0]
            ind2 = max_ind[1]
            ind3 = max_ind[2]
            d1   = np.linalg.norm([tj[i,ind1,0]-tj[i,ind2,0],tj[i,ind1,1]-tj[i,ind2,1]])
            d2   = np.linalg.norm([tj[i,ind1,0]-tj[i,ind3,0],tj[i,ind1,1]-tj[i,ind3,1]])
            d3   = np.linalg.norm([tj[i,ind2,0]-tj[i,ind3,0],tj[i,ind2,1]-tj[i,ind3,1]])
            if d1 > d2 and d1 > d3:
                indexes.append((ind1,ind2))
            elif d2 > d1 and d2 > d3:
                indexes.append((ind1,ind3))
            else:
                indexes.append((ind2,ind3))
            triangles.append((ind1,ind2,ind3))
    
    return indexes, triangles

def extract_indexes_new(tj, tol):

    final_triangles = []
    indexes, triangles = extract_indexes(tj)
    for i in range(tj.shape[0]):
        temp_triangles = list(triangles[i])
        if len(triangles[i]) == 0:
            temp_triangles = []
        else:
            p1 = tj[i,triangles[i][0],:]
            p2 = tj[i,triangles[i][1],:]
            p3 = tj[i,triangles[i][2],:]
            for j in range(tj.shape[1]):
                if np.linalg.norm(p1 - tj[i,j,:]) < tol or np.linalg.norm(p2 - tj[i,j,:]) < tol or np.linalg.norm(p3 - tj[i,j,:]) < tol:
                    if j not in temp_triangles:
                        temp_triangles.append(j)
        
        final_triangles.append(temp_triangles)
        
    return final_triangles

def sliding_window_embedding_batch_1d(x, window_size, delay, step):

    x   = tf.convert_to_tensor(x)
    L   = int(window_size)
    tau = int(delay)
    s   = int(step)
    T   = tf.shape(x)[1]

    win_len   = (L - 1) * tau + 1
    n_windows = (T - win_len) // s + 1

    starts  = tf.range(n_windows) * s             # (n_windows,)
    offsets = tf.range(L) * tau                   # (L,)
    idx     = starts[:, None] + offsets[None, :]  # (n_windows, L)

    emb = tf.gather(x, idx, axis=1)

    return emb


def sliding_window(batch_x, window_size, delay, step):

    batch_x = tf.convert_to_tensor(batch_x)

    x1 = batch_x[:, :, 0]   # shape (B, T)
    x2 = batch_x[:, :, 1]   # shape (B, T)

    emb_state1 = sliding_window_embedding_batch_1d(x1, window_size=window_size, delay=delay, step=step)
    emb_state2 = sliding_window_embedding_batch_1d(x2, window_size=window_size, delay=delay, step=step)

    return emb_state1, emb_state2


def build_topological_index_matrix(selected_indexes):

    B = len(selected_indexes)
    max_len = max(len(item) for item in selected_indexes)

    topo_idx_mat = np.zeros((B, max_len), dtype=np.int64)
    topo_mask_mat = np.zeros((B, max_len), dtype=np.float64)

    for b, item in enumerate(selected_indexes):

        if len(item) != 0:
            item = np.array(item, dtype=np.int64)

            topo_idx_mat[b, :len(item)] = item
            topo_mask_mat[b, :len(item)] = 1.0

    topo_idx_mat = tf.constant(topo_idx_mat, dtype=tf.int64)
    topo_mask_mat = tf.constant(topo_mask_mat, dtype=tf.float64)

    return topo_idx_mat, topo_mask_mat


def add_relative_gaussian_noise(trajectories, noise_level, seed=0, keep_initial_clean=True):

    trajectories = np.asarray(trajectories, dtype=np.float64)
    rng          = np.random.default_rng(seed)
    traj_std     = np.std(trajectories, axis=1, keepdims=True)
    noise        = rng.normal(loc=0.0, scale=1.0, size=trajectories.shape)
    noise        = noise_level * traj_std * noise

    if keep_initial_clean:
        noise[:, 0, :] = 0.0

    return trajectories + noise

def to_numpy(x):

    if hasattr(x, "numpy"):
        return x.numpy()

    return np.asarray(x)


def get_time_axis(dataset):

    return np.asarray(to_numpy(dataset["times"]), dtype=float).reshape(-1)


def denormalize_output_epi(out_fields, problem, normalization_definition):

    out_fields    = np.asarray(to_numpy(out_fields), dtype=float)
    normalization = analyze_normalization_epi(problem, normalization_definition)
    axis          = out_fields.ndim - 1

    return normalize_back(
        out_fields,
        normalization["out_fields_min"],
        normalization["out_fields_max"],
        axis=axis
    )


def trajectory_mse(pred, true):

    pred = np.asarray(to_numpy(pred), dtype=float)
    true = np.asarray(to_numpy(true), dtype=float)

    return np.mean(np.square(pred - true))


def oscillation_features(time, signal, transient_time=10.0, prominence_fraction=0.10):

    time   = np.asarray(time, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)

    mask = (time >= transient_time)
    t    = time[mask]
    y    = signal[mask]

    # Not enough post-transient samples
    if len(y) < 3:
        return np.nan, np.nan, False

    # Invalid signal
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan, False

    signal_range = np.ptp(y)

    # Collapsed or approximately constant trajectory
    if signal_range < 1e-8:
        return np.nan, np.nan, False

    prominence = prominence_fraction * signal_range

    peaks, _   = find_peaks(y, prominence=prominence)
    troughs, _ = find_peaks(-y, prominence=prominence)

    # A valid oscillation must contain at least two maxima
    # and at least one minimum in the considered interval.
    is_periodic = (len(peaks) >= 2 and len(troughs) >= 1)

    if not is_periodic:
        return np.nan, np.nan, False

    # Period
    period = np.mean(np.diff(t[peaks]))

    # Amplitude
    mean_max  = np.mean(y[peaks])
    mean_min  = np.mean(y[troughs])
    amplitude = 0.5 * (mean_max - mean_min)

    return period, amplitude, True


def symmetric_chamfer_distance(X, Y):

    X = np.asarray(to_numpy(X), dtype=float)
    Y = np.asarray(to_numpy(Y), dtype=float)

    # Remove possible non-finite points
    X = X[np.all(np.isfinite(X), axis=1)]
    Y = Y[np.all(np.isfinite(Y), axis=1)]

    # Chamfer distance cannot be computed on empty point clouds
    if len(X) == 0 or len(Y) == 0:
        return np.nan

    tree_X = cKDTree(X)
    tree_Y = cKDTree(Y)

    d_XY, _ = tree_Y.query(X, k=1)
    d_YX, _ = tree_X.query(Y, k=1)

    return np.mean(d_XY**2) + np.mean(d_YX**2)


def compute_test_metrics(
    predictions,
    dataset,
    mi_values,
    problem,
    normalization_definition,
    transient_time=10.0
):

    predictions    = np.asarray(to_numpy(predictions), dtype=float)
    references     = np.asarray(to_numpy(dataset["out_fields"]), dtype=float)
    mi_values      = np.asarray(mi_values, dtype=float).reshape(-1)
    time           = get_time_axis(dataset)
    pred_phys      = denormalize_output_epi(predictions, problem, normalization_definition)
    true_phys      = denormalize_output_epi(references, problem, normalization_definition)
    transient_mask = (time >= transient_time)

    results = []

    for k in range(predictions.shape[0]):

        # MSE
        mse = trajectory_mse(
            predictions[k],
            references[k]
        )

        # PERIOD AND v AMPLITUDE
        period_true, amp_v_true, periodic_v_true = oscillation_features(
            time,
            true_phys[k, :, 0],
            transient_time
        )

        period_pred, amp_v_pred, periodic_v_pred = oscillation_features(
            time,
            pred_phys[k, :, 0],
            transient_time
        )

        # w AMPLITUDE
        _, amp_w_true, periodic_w_true = oscillation_features(
            time,
            true_phys[k, :, 1],
            transient_time
        )

        _, amp_w_pred, periodic_w_pred = oscillation_features(
            time,
            pred_phys[k, :, 1],
            transient_time
        )

        # PERIOD ERROR
        if (
            periodic_v_true
            and periodic_v_pred
            and np.isfinite(period_true)
            and np.isfinite(period_pred)
            and period_true > 0
        ):

            period_error = abs(period_pred - period_true) / period_true

        else:

            period_error = np.nan

        # v AMPLITUDE ERROR
        if (
            periodic_v_true
            and periodic_v_pred
            and np.isfinite(amp_v_true)
            and np.isfinite(amp_v_pred)
            and amp_v_true > 0
        ):

            amplitude_error_v = abs(amp_v_pred - amp_v_true) / amp_v_true

        else:

            amplitude_error_v = np.nan

        # w AMPLITUDE ERROR
        if (
            periodic_w_true
            and periodic_w_pred
            and np.isfinite(amp_w_true)
            and np.isfinite(amp_w_pred)
            and amp_w_true > 0
        ):

            amplitude_error_w = abs(amp_w_pred - amp_w_true) / amp_w_true

        else:

            amplitude_error_w = np.nan

        # PHASE-SPACE ERROR
        if np.sum(transient_mask) > 1:

            phase_space_error = symmetric_chamfer_distance(
                references[k][transient_mask],
                predictions[k][transient_mask]
            )

        else:

            phase_space_error = np.nan

        # STORE RESULTS
        results.append({
            "mi":
                float(mi_values[k]),
            "MSE":
                mse,
            "period_true":
                period_true,
            "period_pred":
                period_pred,
            "period_error":
                period_error,
            "amplitude_v_true":
                amp_v_true,
            "amplitude_v_pred":
                amp_v_pred,
            "amplitude_error_v":
                amplitude_error_v,
            "amplitude_w_true":
                amp_w_true,
            "amplitude_w_pred":
                amp_w_pred,
            "amplitude_error_w":
                amplitude_error_w,
            "phase_space_error":
                phase_space_error,
            "periodic_v_true":
                periodic_v_true,
            "periodic_v_pred":
                periodic_v_pred,
            "periodic_w_true":
                periodic_w_true,
            "periodic_w_pred":
                periodic_w_pred,
        })

    return pd.DataFrame(results)


def metric_summary(df, label):

    metric_names = [
        "MSE",
        "period_error",
        "amplitude_error_v",
        "amplitude_error_w",
        "phase_space_error"
    ]

    rows = []

    for metric in metric_names:

        values = (
            df[metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        n_valid = len(values)
        n_total = len(df)

        if n_valid == 0:

            rows.append({
                "time_step":
                    label,
                "metric":
                    metric,
                "n_valid":
                    0,
                "valid_fraction":
                    0.0,
                "median":
                    np.nan,
                "Q1":
                    np.nan,
                "Q3":
                    np.nan,
                "mean":
                    np.nan,
                "std":
                    np.nan
            })

            continue

        rows.append({
            "time_step":
                label,
            "metric":
                metric,
            "n_valid":
                n_valid,
            "valid_fraction":
                n_valid / n_total,
            "median":
                np.median(values),
            "Q1":
                np.percentile(values, 25),
            "Q3":
                np.percentile(values, 75),
            "mean":
                np.mean(values),
            "std":
                np.std(values)
        })

    return pd.DataFrame(rows)


def periodicity_summary(df, label):

    n_total = len(df)

    n_periodic_v = int(
        df["periodic_v_pred"].sum()
    )

    n_periodic_w = int(
        df["periodic_w_pred"].sum()
    )

    return pd.DataFrame([{
        "time_step":
            label,
        "n_trajectories":
            n_total,
        "n_periodic_v":
            n_periodic_v,
        "periodic_fraction_v":
            n_periodic_v / n_total,
        "n_periodic_w":
            n_periodic_w,
        "periodic_fraction_w":
            n_periodic_w / n_total
    }])