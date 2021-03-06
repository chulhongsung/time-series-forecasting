import numpy as np
import tensorflow as tf
from tensorflow import keras as K

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, d_model, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(d_model, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class CateFeatureEmbedding(K.layers.Layer):
    def __init__(self, d_embedding, cat_dim):
        super(CateFeatureEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.embedding_layer = K.layers.Embedding(input_dim=cat_dim, output_dim=d_embedding)
      
    def call(self, x):
        return tf.expand_dims(self.embedding_layer(x), axis=2)
    
class MultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
    
class AddPosition(K.layers.Layer):
    def __init__(self, d_model, timesteps):
        super(AddPosition, self).__init__()
        self.layer_norm = K.layers.LayerNormalization()
        self.posit_matrix = positional_encoding(timesteps, d_model)
          
    def call(self, x, t):
        return self.layer_norm(x + self.posit_matrix[:, t:t+1 :])
    
class AddPosition2(K.layers.Layer):
    def __init__(self, d_model, timesteps):
        super(AddPosition2, self).__init__()
        self.layer_norm = K.layers.LayerNormalization()
        self.posit_matrix = positional_encoding(timesteps, d_model)
       
    def call(self, x):
        return self.layer_norm(x + self.posit_matrix)
    
class GenLayer(K.layers.Layer):
    def __init__(self, d_model, d_latent, timesteps, num_heads):
        super(GenLayer, self).__init__()
        self.d_model = d_model
        self.timesteps = timesteps
        
        self.w0 = tf.Variable(tf.random.normal([1, 1, d_model])) # (batch_size, timestep, d_model)
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        self.mha4 = MultiHeadAttention(d_model, num_heads)

        self.dense1 = K.layers.Dense(d_latent)
        self.dense2 = K.layers.Dense(d_latent)
        self.dense3 = K.layers.Dense(d_model)
        
        self.layer_norm1 = K.layers.LayerNormalization()
        self.layer_norm2 = K.layers.LayerNormalization()
        self.layer_norm3 = K.layers.LayerNormalization()
        self.layer_norm4 = K.layers.LayerNormalization()

        self.add_posit = AddPosition(d_model, timesteps)
    
    def call(self, h_C, h_prime_T, prior_W=None, MC=False):
        batch_size = h_C.shape[0]
        w = tf.repeat(self.w0, batch_size, axis=0)
        
        z_list = []
        w_hat_list = []
        w_trend_list = []
        mean_list = []
        var_list = []
        
        for i in range(self.timesteps):
            if prior_W == None:
                tmp_w_bar = self.layer_norm1(w[:, i:i+1, :] + self.mha1(w[:, i:i+1, :], w[:, :i+1, :], w[:, :i+1, :]))
            else:
                tmp_w_tilde = self.layer_norm4(w[:, i:i+1, :] + self.mha4(w[:, i:i+1, :], prior_W, prior_W))
                tmp_w_bar = self.layer_norm1(tmp_w_tilde + self.mha1(tmp_w_tilde, w[:, :i+1, :], w[:, :i+1, :]))

            tmp_w_hat = self.layer_norm2(tmp_w_bar + self.mha2(tmp_w_bar, h_C, h_C))
            tmp_w_trend = self.layer_norm3(tmp_w_hat + self.mha3(tmp_w_hat, h_prime_T, h_prime_T)) 
            tmp_mean = self.dense1(tmp_w_hat)
            if MC:
                tmp_eps = tf.zeros_like(tmp_mean, dtype=tf.float32)
                for _ in range(5):
                    tmp_eps += tf.random.normal(shape=tmp_eps.shape,
                                                mean=tf.zeros(shape=tmp_mean.shape),
                                                stddev=tf.math.sigmoid(self.dense2(tmp_w_hat)))/5
            else:
                   tmp_eps = tf.random.normal(shape=tmp_mean.shape,
                                              mean=tf.zeros(shape=tmp_mean.shape),
                                              stddev=tf.math.sigmoid(self.dense2(tmp_w_hat)))
            tmp_z = tmp_mean + tmp_eps    
            tmp_w = self.add_posit(tmp_w_trend + self.dense3(tmp_z), i)

            w = tf.concat([w, tmp_w], axis=1)
            w_trend_list.append(tmp_w_trend)
            w_hat_list.append(tmp_w_hat) 
            z_list.append(tmp_z) 
            mean_list.append(tmp_mean)
            var_list.append(tf.math.sigmoid(self.dense2(tmp_w_hat)))
                           
        z = tf.concat(z_list, axis=1) # (batch_size, timesteps, d_latent)
        w_hat = tf.concat(w_hat_list, axis=1) # (batch_size, timesteps, d_model)
        w_trend = tf.concat(w_trend_list, axis=1) # (batch_size, timesteps, d_model)
        mean = tf.concat(mean_list, axis=1) # (batch_size, timesteps, d_latent)
        var = tf.concat(var_list, axis=1) # (batch_size, timesteps, d_latent)
        
        return w[:, 1:, :], z, w_hat, w_trend, mean, var
    
class InfLayer(K.layers.Layer):
    def __init__(self, d_model, d_latent, num_heads):
        super(InfLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.dense1 = K.layers.Dense(d_latent)
        self.dense2 = K.layers.Dense(d_latent)

    def call(self, h_T, h_prime_T, w_hat, w_trend):
        h = tf.concat([h_T, h_prime_T], axis=-1)
        k = self.mha1(h, h, h)
        tk = K.layers.concatenate([w_trend, k], axis=-1)
        hk = K.layers.concatenate([w_hat, k], axis=-1)
        tmp_mean = self.dense1(tk)
        
        tmp_eps = tf.zeros_like(tmp_mean, dtype=tf.float32)

        tmp_eps = tf.random.normal(shape=tmp_mean.shape,
                                   mean=tf.zeros(shape=tmp_mean.shape),
                                   stddev=tf.math.sigmoid(self.dense2(hk)))
        
        z = tmp_mean + tmp_eps
            
        return z, tmp_mean, tf.math.softplus(self.dense2(hk))
