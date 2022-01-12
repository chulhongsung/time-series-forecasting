import tensorflow as tf
from tensorflow import keras as K

class ContiFeatureEmbedding(K.layers.Layer):
    def __init__(self, d_embedding, num_rv):
        super(ContiFeatureEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.num_rv = num_rv
        self.dense_layers = [K.layers.Dense(d_embedding) for _ in range(num_rv)]
        
    def call(self, x):
        tmp_feature_list = []                    

        for i in range(self.num_rv):
            tmp_feature = self.dense_layers[i](x[:, :, i:i+1])            
            tmp_feature_list.append(tmp_feature)

        feature_list = tf.concat(tmp_feature_list, axis=-1) ### (batch_size, time_step, num_rv * d_embedding)
        
        return tf.reshape(feature_list, [feature_list.shape[0], feature_list.shape[1], self.num_rv, self.d_embedding])

class CateFeatureEmbedding(K.layers.Layer):
    def __init__(self, d_embedding, cat_dim):
        super(CateFeatureEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.embedding_layer = K.layers.Embedding(input_dim=cat_dim, output_dim=d_embedding)
        
    def call(self, x):
        return tf.expand_dims(self.embedding_layer(x), axis=2)

class GLULN(K.layers.Layer):
    def __init__(self, d_model):
        super(GLULN, self).__init__()    
        self.dense1 = K.layers.Dense(d_model, activation='sigmoid')
        self.dense2 = K.layers.Dense(d_model)
        self.layer_norm = K.layers.LayerNormalization()
        
    def call(self, x, y):
        return self.layer_norm(tf.keras.layers.Multiply()([self.dense1(x),
                                        self.dense2(x)]) + y)

class GatedResidualNetwork(K.layers.Layer):
    def __init__(self, d_model, dr): 
        super(GatedResidualNetwork, self).__init__()        
        self.dense1 = K.layers.Dense(d_model, activation='elu')        
        self.dense2 = K.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dr)
        self.glu_and_layer_norm = GLULN(d_model)
        
        
    def call(self, a):
        eta_2 = self.dense1(a)
        eta_1 = self.dropout(self.dense2(eta_2))
        grn_output = self.glu_and_layer_norm(eta_1, eta_2)
        
        return grn_output

class VariableSelectionNetwork(K.layers.Layer):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.dr = dr
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = K.layers.Softmax()
        self.xi_grn = [GatedResidualNetwork(d_model, dr) for _ in range(self.d_input)]
 
    def call(self, xi):
        
        Xi = tf.reshape(xi, [xi.shape[0], xi.shape[1], -1])
        weights = tf.expand_dims(self.softmax(self.v_grn(Xi)), axis=-1)
            
        tmp_xi_list = []                    
        
        for i in range(self.d_input):
            tmp_xi = self.xi_grn[i](xi[:, :, i:i+1, :])            
            tmp_xi_list.append(tmp_xi)
        
        xi_list = tf.concat(tmp_xi_list, axis=2)
        combined = tf.keras.layers.Multiply()([weights, xi_list]) # attention

        vsn_output = tf.reduce_sum(combined, axis=2) 
    
        return vsn_output

def scaled_dot_product_attention(q, k, v, d_model, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(d_model, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # mask_ = mask[:, tf.newaxis, ...] - 1 # (batch_size, num_heads, seq_len_q, seq_len_q)
    # scaled_attention_logits += (mask_ * 1e9)
    attention_weights = K.layers.Softmax()(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class InterpretableMultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(InterpretableMultiHeadAttention, self).__init__()
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

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        scaled_attention_mean = tf.reduce_mean(scaled_attention, axis=2)
        
        concat_attention = tf.reshape(scaled_attention_mean,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TemporalFusionDecoder(K.layers.Layer):
    def __init__(self, d_model, dr, num_heads):
        super(TemporalFusionDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        ### observed feature embedding        
        self.lstm_obs = K.layers.LSTM(d_model,
                                      return_sequences=True,
                                      return_state=True,
                                      activation='tanh',
                                      recurrent_activation='sigmoid',
                                      recurrent_dropout=0,
                                      unroll=False,
                                      use_bias=True)
        
        self.lstm_future = K.layers.LSTM(d_model,
                                return_sequences=True,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True)        

        self.glu_and_layer_norm1= GLULN(d_model)
        
        self.imha = InterpretableMultiHeadAttention(d_model, num_heads=num_heads)
        
        self.glu_and_layer_norm2 = GLULN(d_model)
        
        
    def call(self, vsn_obs_feature, vsn_future_feature):
        time_step = tf.shape(vsn_obs_feature)[1] + tf.shape(vsn_future_feature)[1]
        obs_lstm, obs_h, obs_c = self.lstm_obs(vsn_obs_feature)
        future_lstm = self.lstm_future(vsn_future_feature, initial_state=[obs_h, obs_c])

        lstm_hidden = tf.concat([obs_lstm, future_lstm], axis=1)    
        input_vsn = tf.concat([vsn_obs_feature, vsn_future_feature], axis=1)

        glu_phi_list = [] 
        
        for j in range(time_step):
            tmp_phi_t = self.glu_and_layer_norm1(lstm_hidden[:, j, :], input_vsn[:, j, :])
            glu_phi_list.append(tf.expand_dims(tmp_phi_t, axis=1))

        glu_phi = tf.concat(glu_phi_list, axis=1)

        B, _ = self.imha(glu_phi, glu_phi, glu_phi) # imha output, weights

        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.glu_and_layer_norm2(B[:, j, :], glu_phi[:, j, :])
            glu_delta_list.append(tf.expand_dims(tmp_delta_t, axis=1))

        glu_delta = tf.concat(glu_delta_list, axis=1)

        return glu_delta, glu_phi

class PointWiseFeedForward(K.layers.Layer):
    def __init__(self, d_model, dr):
        super(PointWiseFeedForward, self).__init__()
        self.grn = GatedResidualNetwork(d_model, dr)
        self.glu_and_layer_norm = GLULN(d_model)

        
    def call(self, delta, phi):
        time_step = tf.shape(delta)[1]
        
        grn_varphi_list = []

        for t in range(time_step):
            tmp_grn_varphi = self.grn(delta[:, t, :])
            grn_varphi_list.append(tf.expand_dims(tmp_grn_varphi, axis=1))

        grn_varphi = tf.concat(grn_varphi_list, axis=1)
        
        varphi_tilde_list = []
        
        for t in range(time_step):
            tmp_varphi_tilde_list = self.glu_and_layer_norm(grn_varphi[:, t, :], phi[:, t, :])
            varphi_tilde_list.append(tf.expand_dims(tmp_varphi_tilde_list, axis=1))
            
        varphi = tf.concat(varphi_tilde_list, axis=1)
            
        return varphi

class QuantileOutput(K.layers.Layer):
    def __init__(self, tau, quantile):
        super(QuantileOutput, self).__init__()
        self.tau = tau
        self.quantile = quantile
        self.quantile_dense = [K.layers.Dense(1) for _ in range(len(quantile))]

    def call(self, varphi):
        total_output_list = []
        for j in range(len(self.quantile)):
            tmp_quantile_list = []
            for t in range(self.tau):
                tmp_quantile = self.quantile_dense[j](varphi[:, -self.tau + j, :])
                tmp_quantile_list.append(tf.expand_dims(tmp_quantile, axis=1))
            total_output_list.append(tf.transpose(tf.concat(tmp_quantile_list, axis=1), perm=[0, 2, 1]))

        return tf.concat(total_output_list, axis=1)