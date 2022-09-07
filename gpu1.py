import tensorflow as tf
import numpy as np

a = ['ddd', 'eee', 'fff']
b = ['aaa', 'bbb', 'ccc']
tf.keras.layers.Concatenate()([tf.convert_to_tensor(a), tf.convert_to_tensor(b)])