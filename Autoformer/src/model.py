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
        
        self.projector2 = K.layers.Conv1D(1, kernel_size=3, strides=1, padding='same', use_bias=False)

    @tf.function
    def decomp(self, x):
        timesteps = x.shape[1]
        x_ens, x_ent = self.series_decomp(x[:, timesteps//2:, :])
        x_des = tf.concat([x_ens, tf.zeros([x_ens.shape[0], self.future_steps, x_ens.shape[-1]])], axis=1)
        x_det = tf.concat([x_ent, tf.repeat(tf.reduce_mean(x, axis=1)[:, tf.newaxis, :], repeats=self.future_steps, axis=1)], axis=1)

        for i in range(self.N):
            x = self.encoder[i](x)

        encoder_output = x

        trend_init = x_det[..., 0:1]

        for j in range(self.M):
            x_des, trend_init = self.decoder[j](x_des, encoder_output, trend_init)
        
        trend = trend_init

        return trend[:, -self.future_steps:, :], x_des[:, -self.future_steps:, :]


    @tf.function
    def call(self, x):

        trend, seasonal_ = self.decomp(x)

        seasonal = self.projector2(seasonal_)

        return trend, seasonal