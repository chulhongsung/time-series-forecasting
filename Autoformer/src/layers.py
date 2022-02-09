import math
import tensorflow as tf
from tensorflow import keras as K

class SeriesDecomp(K.layers.Layer):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = K.layers.AveragePooling1D(pool_size=kernel_size, strides=1,padding='valid')
        
    def call(self, x):
        front = tf.tile(x[:, 0:1, :], tf.constant([1, (self.kernel_size - 1)//2, 1]))
        end = tf.tile(x[:, -1:, :], tf.constant([1, (self.kernel_size - 1)//2, 1]))
        x_ = tf.concat([front, x, end], axis=1)
        x_t = self.moving_avg(x_)
        x_s = x - x_t

        return x_s, x_t
    
class AutoCorrelation(K.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = K.layers.Dense(d_model)
        self.wk = K.layers.Dense(d_model)
        self.wv = K.layers.Dense(d_model)

        self.dropout = K.layers.Dropout(attention_dropout)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, timesteps, depth)

    def time_delay_agg(self, q, k, v):
        batch_size, _, timesteps, __ = q.shape
      
        q_fft = tf.signal.rfft(tf.transpose(q, perm=[0, 1, 3, 2])) # transpose: split_q (batch_size, num_heads, timesteps, depth) ->  (batch_size, num_heads, depth, timesteps)
        k_fft = tf.signal.rfft(tf.transpose(k, perm=[0, 1, 3, 2]))     
        
        S_qk = q_fft * tf.math.conj(k_fft)
        
        R_qk = tf.signal.irfft(S_qk)

        init_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(timesteps), axis=0), axis=0), axis=0), tf.constant([batch_size, self.num_heads, self.depth, 1]))

        top_k = int(2 * math.log(timesteps))
        mean_value = tf.reduce_mean(R_qk, axis=1)
        weights, indices = tf.math.top_k(R_qk, top_k)

        tmp_corr = tf.nn.softmax(weights, axis=-1)

        tmp_values = tf.tile(tf.transpose(q, perm=[0, 1, 3, 2]), tf.constant([1, 1, 1, 2]))
        delays_agg = tf.zeros_like(tf.transpose(q, perm=[0, 1, 3, 2]))

        for i in range(top_k):
            pattern = tf.gather(tmp_values, init_index + tf.expand_dims(indices[..., i], -1), axis=-1, batch_dims=-1)
            delays_agg = delays_agg + pattern * (tf.expand_dims(tmp_corr[..., i], axis=-1))  

        return delays_agg

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.dropout(self.wq(q))  # (batch_size, timesteps, d_model)
        k = self.dropout(self.wk(k))  # (batch_size, timesteps, d_model)
        v = self.dropout(self.wv(v))  # (batch_size, timesteps, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, timesteps_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, timesteps_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, timesteps_v, depth)

        B, H, L, E = q.shape
        _, _, S, D = v.shape
        
        if L > S:
          zeros = tf.zeros_like(q[:, :, :(L - S), :])
          v = tf.concat([v, zeros], axis=2)
          k = tf.concat([k, zeros], axis=2)
        else:
          v = v[:, :, :L, :]
          k = k[:, :, :L, :]
        
        delays_agg_ = self.time_delay_agg(q, k, v)

        delays_agg = tf.transpose(delays_agg_, perm=[0, 3, 1, 2]) # (batch_size, timesteps, num_heads, depth)

        concat_delays_agg = tf.reshape(delays_agg, (batch_size, -1, self.d_model))  # (batch_size, timesteps, d_model)

        output = self.dense(concat_delays_agg) 

        return output
    
class EncoderLayer(K.layers.Layer):
    def __init__(self, kernel_size, d_model, num_heads, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.series_decomp1 = SeriesDecomp(kernel_size)
        self.series_decomp2 = SeriesDecomp(kernel_size)
        self.autocorrelation = AutoCorrelation(d_model, num_heads)
        
        self.dense1 = K.layers.Dense(d_model)
        self.dense2 = K.layers.Dense(d_model)

        self.dropout = K.layers.Dropout(dropout_rate)
        
    def call(self, x):
        x = self.dropout(self.dense1(x))
        x, _ = self.series_decomp1(self.autocorrelation(x, x, x) + x)
        x, _ = self.series_decomp2(self.dropout(self.dense2(x)) + x)

        return x
    
class DecoderLayer(K.layers.Layer):
    def __init__(self, kernel_size, d_model, num_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.series_decomp1 = SeriesDecomp(kernel_size)
        self.series_decomp2 = SeriesDecomp(kernel_size)
        self.series_decomp3 = SeriesDecomp(kernel_size)
        
        self.autocorrelation1 = AutoCorrelation(d_model, num_heads)
        self.autocorrelation2 = AutoCorrelation(d_model, num_heads)
        
        self.dense1 = K.layers.Dense(d_model)
        self.dense2 = K.layers.Dense(d_model)
        self.projector = K.layers.Dense(1, use_bias=False)
        self.dropout = K.layers.Dropout(dropout_rate)

    def call(self, x, x_en, init_trend):
        x = self.dropout(self.dense1(x))
        x, trend1 = self.series_decomp1(self.autocorrelation1(x, x, x) + x)
        x, trend2 = self.series_decomp2(self.autocorrelation2(x, x_en, x_en) + x)
        x, trend3 = self.series_decomp3(self.dropout(self.dense2(x)) + x)
        
        trend = tf.concat([trend1,trend2,trend3], axis=-1)
        trend = self.dropout(self.projector(trend))

        return x, init_trend+trend
