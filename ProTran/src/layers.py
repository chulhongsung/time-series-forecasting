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

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

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
        
        self.dense1 = K.layers.Dense(d_latent)
        self.dense2 = K.layers.Dense(d_latent)
        self.dense3 = K.layers.Dense(d_model)
        
        self.layer_norm1 = K.layers.LayerNormalization()
        self.layer_norm2 = K.layers.LayerNormalization()
        self.layer_norm3 = K.layers.LayerNormalization()
        
        self.add_posit = AddPosition(d_model, timesteps)
    
    def call(self, h_C, prior_W=None):
        batch_size = h_C.shape[0]
        w = tf.repeat(self.w0, batch_size, axis=0)
        
        z_list = []
        w_hat_list = []
        mean_list = []
        var_list = []
        
        for i in range(self.timesteps):
            if prior_W == None:
                tmp_w_bar = self.layer_norm1(w[:, i:i+1, :] + self.mha1(w[:, i:i+1, :], w[:, :i+1, :], w[:, :i+1, :]))
                tmp_w_hat = self.layer_norm2(tmp_w_bar + self.mha2(tmp_w_bar, h_C, h_C)) 
                tmp_mean = self.dense1(tmp_w_hat)
                tmp_eps = tf.random.normal(shape=tmp_mean.shape, mean=tf.zeros(shape=tmp_mean.shape), stddev=tf.math.softplus(self.dense2(tmp_w_hat)))
                tmp_z = tmp_mean + tmp_eps    
                tmp_w = self.add_posit(tmp_w_hat + self.dense3(tmp_z), i)
               
                w = tf.concat([w, tmp_w], axis=1) 
                
                w_hat_list.append(tmp_w_hat) 
                z_list.append(tmp_z) 
                mean_list.append(tmp_mean)
                var_list.append(tf.math.softplus(self.dense2(tmp_w_hat)))
                
            else:
                tmp_w_tilde = self.layer_norm3(w[:, i:i+1, :] + self.mha3(w[:, i:i+1, :], prior_W, prior_W))
                tmp_w_bar = self.layer_norm1(tmp_w_tilde + self.mha1(tmp_w_tilde, w[:, :i+1, :], w[:, :i+1, :]))
                tmp_w_hat = self.layer_norm2(tmp_w_bar + self.mha2(tmp_w_bar, h_C, h_C)) 
                tmp_mean = self.dense1(tmp_w_hat)
                tmp_eps = tf.random.normal(shape=tmp_mean.shape, mean=tf.zeros(shape=tmp_mean.shape), stddev=tf.math.softplus(self.dense2(tmp_w_hat)))
                tmp_z = tmp_mean + tmp_eps    
                tmp_w = self.add_posit(tmp_w_hat + self.dense3(tmp_z), i)

                w = tf.concat([w, tmp_w], axis=1)

                w_hat_list.append(tmp_w_hat) 
                z_list.append(tmp_z) 
                mean_list.append(tmp_mean)
                var_list.append(tf.math.softplus(self.dense2(tmp_w_hat)))
                           
        z = tf.concat(z_list, axis=1) # (batch_size, timesteps, d_latent)
        w_hat = tf.concat(w_hat_list, axis=1) # (batch_size, timesteps, d_model)
        mean = tf.concat(mean_list, axis=1) # (batch_size, timesteps, d_latent)
        var = tf.concat(var_list, axis=1) # (batch_size, timesteps, d_latent)
        
        return w[:, 1:, :], z, w_hat, mean, var

class InfLayer(K.layers.Layer):
    def __init__(self, d_model, d_latent, num_heads):
        super(InfLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.dense1 = K.layers.Dense(d_latent)
        self.dense2 = K.layers.Dense(d_latent)
            
    def call(self, h_T, w_hat):
        k = self.mha1(h_T, h_T, h_T)
        hw = K.layers.concatenate([w_hat, k], axis=-1)
        mean = self.dense1(hw)
        eps = tf.random.normal(shape=mean.shape, mean=tf.zeros(shape=mean.shape), stddev=tf.math.softplus(self.dense2(hw)))
        z = mean + eps
            
        return z, mean, tf.math.softplus(self.dense2(hw))

class ProTran(K.models.Model):
    def __init__(self, d_output, d_model, d_latent, timesteps, current_time, num_heads, num_layers):
        super(ProTran, self).__init__()   
        self.d_model = d_model
        self.timesteps = timesteps
        self.current_time = current_time
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.dense = K.layers.Dense(d_model)
        self.add_posit = AddPosition2(d_model, timesteps)
        
        self.gen_decoder = [GenLayer(d_model, d_latent, timesteps, num_heads) for _ in range(num_layers)]
        self.inf_encoder = [InfLayer(d_model, d_latent, num_heads) for _ in range(num_layers)]
        
        self.final_dense = K.layers.Dense(d_output)
        
    def call(self, x):
        h = self.add_posit(self.dense(x))
        
        gen_w_list = []        
        gen_z_list = []
        gen_w_hat_list = []                    
        gen_mean_list = []
        gen_var_list = []
                    
        for i in range(self.num_layers):
            if i == 0 :
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :])
            else:
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :], tmp_gen_w)
            
            gen_w_list.append(tf.expand_dims(tmp_gen_w, axis=1)) # (batch_size, 1, timesteps, d_model)
            gen_z_list.append(tf.expand_dims(tmp_gen_z, axis=1)) # (batch_size, 1, timesteps, d_latent)
            gen_w_hat_list.append(tf.expand_dims(tmp_gen_w_hat, axis=1)) # (batch_size, 1, timesteps, d_model)
            gen_mean_list.append(tf.expand_dims(tmp_gen_mean, axis=1)) # (batch_size, 1, timesteps, d_latent)
            gen_var_list.append(tf.expand_dims(tmp_gen_var, axis=1)) # (batch_size, 1, timesteps, d_latent)
            
        gen_w = tf.concat(gen_w_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        gen_z = tf.concat(gen_z_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_w_hat = tf.concat(gen_w_hat_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        gen_mean = tf.concat(gen_mean_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_var = tf.concat(gen_var_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        
        x_hat = self.final_dense(gen_w) # (batch_size, num_layers, timesteps, 1)
        
        inf_z_list = []
        inf_mean_list = []
        inf_var_list = []
        
        for i in range(self.num_layers):
            tmp_inf_z, tmp_inf_mean, tmp_inf_var = self.inf_encoder[i](h, gen_w_hat[:, i, :])
            inf_z_list.append(tf.expand_dims(tmp_inf_z, axis=1)) # (batch_size, 1, timesteps, d_latent)
            inf_mean_list.append(tf.expand_dims(tmp_inf_mean, axis=1)) # (batch_size, 1, timesteps, d_latent)
            inf_var_list.append(tf.expand_dims(tmp_inf_var, axis=1)) # (batch_size, 1, timesteps, d_latent)
        
        inf_z = tf.concat(inf_z_list, axis=1)
        inf_mean = tf.concat(inf_mean_list, axis=1)
        inf_var = tf.concat(inf_var_list, axis=1)
        
        return x_hat, gen_z, gen_mean, gen_var, inf_z, inf_mean, inf_var
