import tensorflow as tf
from tensorflow import keras as K
from layers import *

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
    
    @tf.function
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
