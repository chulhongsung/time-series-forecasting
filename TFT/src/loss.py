import tensorflow as tf
from tensorflow import keras as K

class QuantileRisk(K.losses.Loss):
    def __init__(self, tau, quantile):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile 
        self.q_arr = tf.repeat(tf.expand_dims(quantile, axis=0)[..., tf.newaxis], tau, axis=-1)
                
    def call(self, true, pred):
        true_rep = tf.repeat(true, len(self.quantile), axis=1)
        ql = tf.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr) * (pred - true_rep) )
        
        return tf.reduce_mean(ql)