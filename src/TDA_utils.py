import tensorflow as tf

def extract_distance_matrix(states):
    # Dimensions expansion for broadcasting
    diff = tf.expand_dims(states, 2) - tf.expand_dims(states, 1)  # (n_samples, n_times, n_times, n_latent_states)

    # Distance matrices computation
    dist_matrix = tf.norm(diff, axis=-1)  # (n_samples, n_times, n_times)

    return dist_matrix
