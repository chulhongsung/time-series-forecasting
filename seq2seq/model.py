import tensorflow as tf
from tensorflow import keras as K

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
    
    @tf.function
    def call(self, input, train=False, teacher=None):
        input_arr, initial = input
        encoder_hidden, encoder_cell = self.encoder(input_arr)
        output = self.decoder(initial, encoder_hidden, encoder_cell, train, teacher)

        return output
