import tensorflow as tf
from tensorflow import keras as K
from layers import ContiFeatureEmbedding, CateFeatureEmbedding, VariableSelectionNetwork, TemporalFusionDecoder, PointWiseFeedForward, QuantileOutput

class TFT(K.models.Model):
    def __init__(self, d_embedding, num_rv, d_model, dr, cat_dim, future_dim, tau, num_heads):
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
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.qo = QuantileOutput(tau, quantile=[0.1, 0.5, 0.9])
        
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
        varphi = self.pwff(delta, glu_phi)
        
        output = self.qo(varphi)
        
        return output ### (batch_size, quantile, tau)
