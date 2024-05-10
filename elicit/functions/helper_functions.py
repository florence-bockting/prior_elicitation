import pickle
import os
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def save_as_pkl(variable, path_to_file): 
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, 'wb') as df_file:
        pickle.dump(variable, file = df_file) 

class LogsInfo:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    def __call__(self, message, depth):
        if self.max_depth < depth:
            pass
        else: 
            print(message)

def get_lower_triangular(matrix):
    mask = tf.cast(tf.experimental.numpy.tri(matrix.shape[-1], matrix.shape[-1], k=-1), 
                   tf.bool)
    mask_broadcasted = tf.broadcast_to(mask, tuple(matrix.shape))
    lower_triangular = tf.boolean_mask(matrix, mask_broadcasted)
    return lower_triangular