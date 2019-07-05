import tensorflow as tf
from keras import backend
import numpy as np

def GPU_Limit(setMemoryPer=0.5):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = setMemoryPer
    sess = tf.Session(config=config)
    backend.set_session(sess)
    
def GPU_LimitAllow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    backend.set_session(sess)
