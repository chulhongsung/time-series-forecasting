#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import pandas as pd

class Encoder(K.layers.Layer):
    def __init__(self, d_model, dropout=0):
        super(Encoder, self).__init__()        
        self.lstm = K.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout)                
    def call(self, x):
        _, lst_hidden, lst_cell_state  = self.lstm(x)
        
        return lst_hidden, lst_cell_state

class Decoder(K.layers.Layer):
    def __init__(self, d_model, k, dropout=0, activation='sigmoid'): 
        super(Decoder, self).__init__()
        self.k = k

        self.decoder_lstm = K.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout)
        
        self.dense = K.layers.Dense(1, activation=activation) # 1: output dimension
        
    def call(self, initial, encoder_hidden, encoder_cell, train=False, teacher=None): 
        hidden_x, _, __  = self.decoder_lstm(initial, initial_state=[encoder_hidden, encoder_cell])
        hidden_x = self.dense(hidden_x)
        output = [hidden_x] 
        
        if(train):
            # teacher forcing          
            for i in range(self.k-1):
                hidden_x, _, __ = self.decoder_lstm(teacher[:,:,i][:, tf.newaxis, :], initial_state = [_, __])
                hidden_x = self.dense(hidden_x)
                output.append(hidden_x)    
        else:
            for i in range(self.k-1):
                hidden_x, _, __ = self.decoder_lstm(hidden_x, initial_state = [ _, __ ]) 
                hidden_x = self.dense(hidden_x)
                output.append(hidden_x)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(output), perm=[1, 0, 2, 3]))


class Seq2seq(K.models.Model):
    def __init__(self, d_model, k, dropout=0):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(d_model, dropout)
        self.decoder = Decoder(d_model, k, dropout)
        
    def call(self, input, train=False, teacher=None):
        input_arr, initial = input
        encoder_hidden, encoder_cell = self.encoder(input_arr)
        output = self.decoder(initial, encoder_hidden, encoder_cell, train, teacher)

        return output

df_stock = pd.read_csv('stock_sample.csv', index_col=0)

#%% Function preprocess 
def preprocess(data, t, k):
    n = data.shape[0] - t - k
    input_arr = np.zeros((n, t, 5), dtype=np.float32)
    
    target = np.zeros((n, 1, k), dtype=np.float32)
    initial = np.zeros((n, 1, 1), dtype=np.float32)
    
    for i in range(n):
        input_arr[i, :, 0:5] = data.iloc[i:(t+i), 0:5]
        target[i, :, :] = data.iloc[(t+i):(t+i+k), 0]
        initial[i, :, :] = data.iloc[(t+i-1), 0]
        
    return input_arr, initial, target,  
# %%
input_arr, init, target = preprocess(df_stock, 4, 3)

input_arr.shape

train_input = input_arr[:130]
train_init = init[:130]
train_target = target[:130]

test_input = input_arr[130:]
test_init = init[130:]
test_target = target[130:]
#%%
seq2seq = Seq2seq(30, 3, 0.1)

optimizer = K.optimizers.Adam(lr=0.001)
mse = K.losses.MeanSquaredError()

MAX_ITER = 100
#%%
for i in range(MAX_ITER):
    # optimize
    with tf.GradientTape() as tape:
        output = seq2seq([train_input, train_init], train=True, teacher=train_target)
        loss = mse(tf.squeeze(train_target), output)
    grad = tape.gradient(loss, seq2seq.weights)
    optimizer.apply_gradients(zip(grad, seq2seq.weights))       
    
    if (i+1) % 10 == 0:
        print('epoch {:3d}:: TRAIN LOSS : {:.03f}'.format(i+1, loss))  