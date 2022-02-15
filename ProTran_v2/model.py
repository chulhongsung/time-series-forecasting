import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from layers import *

class ProTran(K.models.Model):
    def __init__(self, d_embedding, cat_dim, d_model, d_latent, current_time, num_heads, num_layers):
        super(ProTran, self).__init__()   
        self.d_model = d_model
        self.d_latent = d_latent
        self.current_time = current_time
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cat_dim = cat_dim

        self.dense1 = K.layers.Dense(d_model)
        self.dense2 = K.layers.Dense(d_model)
        self.cfe = [CateFeatureEmbedding(d_embedding, x) for x in cat_dim]
        
        self.inf_encoder = [InfLayer(d_model, d_latent, num_heads) for _ in range(num_layers)]
        

    def build(self, input_shape):
        x_shape, x_future_shape = input_shape

        self.add_posit1 = AddPosition2(self.d_model, x_shape[1])
        self.add_posit2 = AddPosition2(self.d_model, x_shape[1])
        
        self.gen_decoder = [GenLayer(self.d_model, self.d_latent, x_shape[1], self.num_heads) for _ in range(self.num_layers)]
        
        self.final_dense = K.layers.Dense(x_shape[-1])
    
    def generate(self, inputs):
        x, x_future = inputs[0], inputs[1]
        h = self.add_posit1(self.dense1(x))

        tmp_cate_list = [] 

        for k in range(len(self.cat_dim)):
            tmp_cate_input_ = self.cfe[k](x_future[:, :, k])
            tmp_cate_list.append(tmp_cate_input_)

        cate_embedding = tf.squeeze(tf.concat(tmp_cate_list, axis=-1))

        h_prime = self.add_posit2(self.dense2(cate_embedding))

        gen_w_list = []        
                    
        for i in range(self.num_layers):
            if i == 0 :
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_w_trend, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :],
                                                                                                                      h_prime,
                                                                                                                      prior_W=None,
                                                                                                                      MC=True)
            else:
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_w_trend, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :],
                                                                                                                      h_prime, 
                                                                                                                      tmp_gen_w, 
                                                                                                                      MC=True)
            
            gen_w_list.append(tf.expand_dims(tmp_gen_w, axis=1)) # (batch_size, 1, timesteps, d_model)
            
        gen_w = tf.concat(gen_w_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        
        x_hat = self.final_dense(gen_w[:, -1, :, :]) # (batch_size, num_layers, timesteps, 1)

        return x_hat

    @tf.function
    def call(self, inputs, MonteCarlo=True):
        x, x_future = inputs[0], inputs[1]
        h = self.add_posit1(self.dense1(x))

        tmp_cate_list = [] 

        for k in range(len(self.cat_dim)):
            tmp_cate_input_ = self.cfe[k](x_future[:, :, k])
            tmp_cate_list.append(tmp_cate_input_)

        cate_embedding = tf.squeeze(tf.concat(tmp_cate_list, axis=-1))

        h_prime = self.add_posit2(self.dense2(cate_embedding))

        gen_w_list = []        
        gen_z_list = []
        gen_w_hat_list = []
        gen_w_trend_list = []                      
        gen_mean_list = []
        gen_var_list = []
                    
        for i in range(self.num_layers):
            if i == 0 :
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_w_trend, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :],
                                                                                                                      h_prime,
                                                                                                                      prior_W=None,
                                                                                                                      MC=MonteCarlo)
            else:
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_w_trend, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :],
                                                                                                                      h_prime,
                                                                                                                      tmp_gen_w,
                                                                                                                      MC=MonteCarlo)
            
            gen_w_list.append(tf.expand_dims(tmp_gen_w, axis=1)) # (batch_size, 1, timesteps, d_model)
            gen_z_list.append(tf.expand_dims(tmp_gen_z, axis=1)) # (batch_size, 1, timesteps, d_latent)
            gen_w_hat_list.append(tf.expand_dims(tmp_gen_w_hat, axis=1)) # (batch_size, 1, timesteps, d_model)
            gen_w_trend_list.append(tf.expand_dims(tmp_gen_w_trend, axis=1)) # (batch_size, 1, timesteps, d_model)
            gen_mean_list.append(tf.expand_dims(tmp_gen_mean, axis=1)) # (batch_size, 1, timesteps, d_latent)
            gen_var_list.append(tf.expand_dims(tmp_gen_var, axis=1)) # (batch_size, 1, timesteps, d_latent)
            
        gen_w = tf.concat(gen_w_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        gen_z = tf.concat(gen_z_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_w_hat = tf.concat(gen_w_hat_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        gen_w_trend = tf.concat(gen_w_trend_list, axis=1) # (batch_size, num_layers, timesteps, d_model)
        gen_mean = tf.concat(gen_mean_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_var = tf.concat(gen_var_list, axis=1) # (batch_size, num_layers, timesteps, d_latent)
        
        x_hat = self.final_dense(gen_w[:, -1, :, :]) # (batch_size, num_layers, timesteps, d_output)
        
        inf_z_list = []
        inf_mean_list = []
        inf_var_list = []
        
        for i in range(self.num_layers):
            tmp_inf_z, tmp_inf_mean, tmp_inf_var = self.inf_encoder[i](h,
                                                                       h_prime,
                                                                       gen_w_hat[:, i, :],
                                                                       gen_w_trend[:, i, :])
            inf_z_list.append(tf.expand_dims(tmp_inf_z, axis=1)) # (batch_size, 1, timesteps, d_latent)
            inf_mean_list.append(tf.expand_dims(tmp_inf_mean, axis=1)) # (batch_size, 1, timesteps, d_latent)
            inf_var_list.append(tf.expand_dims(tmp_inf_var, axis=1)) # (batch_size, 1, timesteps, d_latent)
        
        inf_z = tf.concat(inf_z_list, axis=1)
        inf_mean = tf.concat(inf_mean_list, axis=1)
        inf_var = tf.concat(inf_var_list, axis=1)
        
        return x_hat, gen_z, gen_mean, gen_var, inf_z, inf_mean, inf_var
