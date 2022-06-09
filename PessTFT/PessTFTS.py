import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import scipy

import argparse 

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=1000, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--tau', required=False, default=70, type=int)
parser.add_argument('--k', required=False, default=200, type=int)
parser.add_argument('--d_emb', required=False, default=3, type=int)
parser.add_argument('--d_model', required=False, default=20, type=int)
parser.add_argument('--dr', required=False, default=0.01, type=float)
parser.add_argument('--num_h', required=False, default=1, type=int)
parser.add_argument('--job_name', required=True, type=str)
parser.add_argument('--load', required=False, default=None, type=str, help="load_model_name")
parser.add_argument('--lambda_1', required=False, default=0.5, type=float)
parser.add_argument('--lambda_2', required=False, default=0.5, type=float)

args = parser.parse_args()

cate_input = np.load("./data/cate_input_k" + str(args.k) + "_tau" + str(args.tau) + ".npy")
real_input = np.load("./data/real_input_k"+ str(args.k) + "_tau" + str(args.tau) + ".npy")
future_input = np.load("./data/future_k" + str(args.k) + "_tau" + str(args.tau) + ".npy")
target = np.load("./data/target_k" + str(args.k) + "_tau" + str(args.tau) + ".npy")

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
    def __init__(self, d_model, d_param, tau, dr):
        super(PointWiseFeedForward, self).__init__()
        self.tau = tau
        self.grn = GatedResidualNetwork(d_model, dr)
        self.glu_and_layer_norm = GLULN(d_model)
        self.beta_map = K.layers.Dense(d_param, activation='softplus')
        self.gamma_map = K.layers.Dense(1)    
        
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
        
        tmp_varphi = varphi[:, -self.tau:, :]
        
        beta = self.beta_map(tmp_varphi)
        gamma = self.gamma_map(tmp_varphi)

        return gamma, beta

class TFT(K.models.Model):
    def __init__(self, d_embedding, num_rv, d_model, d_param, dr, cat_dim, future_dim, tau, num_heads, lambda_1, lambda_2):
        super(TFT, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.cat_dim = cat_dim
        self.d_input = num_rv + len(cat_dim)
        self.future_dim = future_dim
        self.confe = ContiFeatureEmbedding(d_embedding, num_rv)
        self.cfe = [CateFeatureEmbedding(d_embedding, x) for x in cat_dim]
        self.vsn1 = VariableSelectionNetwork(d_model, self.d_input, dr) ### past input
        self.vsn2 = VariableSelectionNetwork(d_model, future_dim, dr) ### future input
        self.tfd = TemporalFusionDecoder(d_model, dr, num_heads)
        self.pwff = PointWiseFeedForward(d_model, d_param, tau, dr)
        self.sodm = tf.constant(scipy.sparse.diags([1., -2., 1.], range(3), shape=(68, 70)).toarray(), tf.float32)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2 
        
    def call(self, rv_inputs, cate_inputs, future_inputs):
      
        conti_embedding = self.confe(rv_inputs)

        tmp_cate_list = [] 
        
        for k in range(len(self.cat_dim)):
            tmp_cate_input_ = self.cfe[k](cate_inputs[:, k, :])
            tmp_cate_list.append(tmp_cate_input_)

        cate_embedding = tf.concat(tmp_cate_list, axis=-2)
        
        obs_feature = tf.concat([conti_embedding, cate_embedding], axis=-2)
        
        tmp_future_list = []
        
        for j in range(self.future_dim):
            tmp_future_input_ = self.cfe[j+2](future_inputs[:, j, :])
            tmp_future_list.append(tmp_future_input_)
        
        future_embedding = tf.concat(tmp_future_list, axis=-2)
        
        x1 = self.vsn1(obs_feature)
        x2 = self.vsn2(future_embedding)
        
        delta, glu_phi = self.tfd(x1, x2)
        gamma, beta = self.pwff(delta, glu_phi)
        smoothing_loss1 = self.lambda_1 * tf.math.reduce_mean(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.linalg.matmul(tf.transpose(beta, [0, 2, 1]), self.sodm, transpose_b=True)), axis=-1)))
        smoothing_loss2 = self.lambda_2 * tf.math.reduce_mean(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.linalg.matmul(tf.transpose(gamma, [0, 2, 1]), self.sodm, transpose_b=True)), axis=-1)))
        self.add_loss(smoothing_loss1 + smoothing_loss2)
        # output = self.qo(varphi)
        
        return gamma, beta 

def knot_value(args):
    """
    Linear isotonic regression spline function 
    
    Compute Knot-points values
    """
    gamma, beta = args
    
    delta = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)

    bl = tf.concat([tf.expand_dims(beta[...,0], axis=-1), (beta[..., 1:] - beta[..., :-1])], axis=-1)

    dl = tf.math.cumsum(delta)
    
    a = tf.math.cumsum(bl * dl, axis=-1)

    b = tf.math.cumsum(bl, axis=-1) * dl

    z = b - a + gamma
    
    return z

def upr(args):
    
    z, gamma, beta, quantile_dl = args 
    
    delta = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)

    mask = tf.cast(z >= quantile_dl, tf.float32)
        
    bl = tf.concat([tf.expand_dims(beta[...,0], axis=-1), (beta[..., 1:] - beta[..., :-1])], axis=-1)

    dl = tf.math.cumsum(delta)
    
    tilde_a = tf.clip_by_value((z - gamma + tf.reduce_sum(bl * dl * mask, axis=-1)[..., tf.newaxis])/ (tf.reduce_sum(bl * mask, axis=-1)[..., tf.newaxis] +0.000001), clip_value_min=0.0001, clip_value_max=1)
    
    upr = (1+tf.math.log(tilde_a))*(z-gamma) + tf.math.reduce_sum(bl*(((-1/2) * ((1-dl)**2)) + 1 - tf.maximum(tilde_a, dl) + dl * tf.maximum(tf.math.log(tilde_a), tf.math.log(dl))), axis=-1)[..., tf.newaxis]
    
    return tf.math.reduce_mean(upr)

train_loss = K.metrics.Mean(name='train_loss')

def train_step(model, real_input, cate_input, future_input, y):
    with tf.GradientTape() as tape:
        tmp_gamma, tmp_beta = model(real_input, cate_input, future_input)
        tmp_quantile_dl = tf.vectorized_map(knot_value, (tmp_gamma, tmp_beta))
        tmp_upr = tf.math.reduce_mean(tf.vectorized_map(upr, (tf.transpose(y, perm=[0, 2, 1]), tmp_gamma, tmp_beta, tmp_quantile_dl)))
    
    grad = tape.gradient(tmp_upr, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    train_loss(tmp_upr)

optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr)

sample_data = tf.data.Dataset.from_tensor_slices((real_input, cate_input, future_input, target))

tft_model = TFT(d_embedding=args.d_emb,
                num_rv=17,
                d_model=args.d_model,
                d_param=11,
                dr=args.dr,
                cat_dim=[4, 4, 12, 31],
                future_dim=2,
                tau=args.tau,
                num_heads=args.num_h,
                lambda_1 = args.lambda_1,
                lambda_2 = args.lambda_2)

BATCH_SIZE = 400

train = sample_data.take(2400)

test = sample_data.skip(2400)

batch_data = train.batch(BATCH_SIZE)

if args.load != None:
    tmp_real, tmp_cate, tmp_future, tmp_true = next(iter(batch_data))
    _, _ = tft_model(tmp_real, tmp_cate, tmp_future)
    tft_model.load_weights("./save_weights/tft_model_" + args.load + ".h5")

EPOCHS = args.epochs

for epoch in tf.range(EPOCHS):
    for tmp_real, tmp_cate, tmp_future, tmp_true in iter(batch_data):
        train_step(tft_model, tmp_real, tmp_cate, tmp_future, tmp_true)
    
    if ((epoch.numpy() + 1) % 100) == 0:
        template = 'EPOCH: {}, Train Loss: {}'
        print(template.format(epoch.numpy()+1, train_loss.result()))
    
tft_model.save_weights("./save_weights/tft_model_" + args.job_name + ".h5")

hyperparam_dict = {"d_emb":args.d_emb,
                   "d_model":args.d_model,
                   "dr": args.dr, 
                   "tau": args.tau,  
                   "num_h": args.num_h, 
                   "lr":args.lr,
                   "epochs":args.epochs,
                   "lambda_1": args.lambda_1,
                   "lambda_2": args.lambda_2}

with open("./param/"+ args.job_name + '_hyperparams.txt', "w") as f:
    print(hyperparam_dict, file=f)
