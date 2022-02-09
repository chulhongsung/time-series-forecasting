import tensorflow as tf
from tensorflow import keras as K
from layers import *

class Autoformer(K.models.Model):
    def __init__(self, N, M, kernel_size, d_model, num_heads, future_steps, dropout_rate=0.1):
        super(Autoformer, self).__init__()
        self.N = N
        self.M = M
        self.future_steps = future_steps

        self.series_decomp = SeriesDecomp(kernel_size)
        self.encoder = [EncoderLayer(kernel_size, d_model, num_heads, dropout_rate) for _ in range(N)]
        self.decoder = [DecoderLayer(kernel_size, d_model, num_heads, dropout_rate) for _ in range(M)]
        self.dense1 = K.layers.Dense(d_model)
        self.dense2 = K.layers.Dense(1)
        self.dense3 = K.layers.Dense(1)
        self.dense4 = K.layers.Dense(future_steps)
        self.dense5 = K.layers.Dense(future_steps)

    @tf.function
    def call(self, x):
        timesteps = x.shape[1]
        x_ens, x_ent = self.series_decomp(x[:, timesteps//2:, :])
        x_des = tf.concat([x_ens, tf.zeros([x_ens.shape[0], self.future_steps, x_ens.shape[-1]])], axis=1)
        x_det = tf.concat([x_ent, tf.repeat(tf.reduce_mean(x, axis=1)[:, tf.newaxis, :], repeats=self.future_steps, axis=1)], axis=1)
        x_det = self.dense1(x_det)
        
        for i in range(self.N):
            x = self.encoder[i](x)

        encoder_output = x

        for j in range(self.M):
            x_des, x_det = self.decoder[j](x_des, encoder_output, x_det)
            
        output = self.dense5(tf.squeeze(self.dense2(x_des))) + self.dense5(tf.squeeze(self.dense3(x_det)))

        return output