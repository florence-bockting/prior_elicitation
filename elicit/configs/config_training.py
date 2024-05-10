import tensorflow as tf
import keras 

def exp_decay_schedule(global_dict):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = global_dict["init_learning_rate"], 
            decay_steps = global_dict["learning_rate_step"], 
            decay_rate = global_dict["learning_rate_percentage"],
            staircase = True)
    return lr_schedule

def cos_decay_schedule(global_dict):
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
             global_dict["init_learning_rate"],
             first_decay_steps = 10, 
             alpha = global_dict["learning_rate_minimum"]
             )
    return lr_schedule