import torch
import torch.nn as nn

from layer import *

class TemporalFusionTransfomer(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        num_cont, 
        num_cate,
        num_stat,
        cate_dims,
        stat_dims,
        d_model,
        seq_len,
        tau,
        quantile,
        dr,
        device
    ):
        super(TemporalFusionTransfomer, self).__init__()
        self.tau = tau
        
        self.confe = ContiFeatureEmbedding(d_embedding, num_cont)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        self.stafe = CateFeatureEmbedding(d_embedding, stat_dims)
        
        self.grn_s = GatedResidualNetwork(d_model, dr)
        self.vsn = VariableSelectionNetwork(d_model, num_cont+num_cate, dr)
        self.vsn_s = VariableSelectionNetwork(d_model, num_stat, dr)
        self.vsn_future = VariableSelectionNetwork(d_model, num_cate, dr)
        
        self.lstm_encoder = nn.LSTM(d_model, d_model, 1, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, 1, batch_first=True)
        
        self.gal1 = GAL(d_model)
        self.gal2 = GAL(d_model)
        self.gal3 = GAL(d_model)
        
        self.static_grn = GatedResidualNetwork(d_model, dr)
        self.mha = nn.MultiheadAttention(d_model, 1, batch_first=True, dropout=dr)
        self.temporal_mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1).to(device)
        self.final_grn = GatedResidualNetwork(d_model, dr)
        self.quantile_layer = QuantileOutput(tau, quantile)
        
        
    def forward(self, conti, cate, static, future):
        confe_output = self.confe(conti) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate)  # (batch_size, seq_len, num_cate, d_embedding)
        stafe_output = self.stafe(static) # (batch_size, 1, num_cate, d_embedding)
        futfe_output = self.catfe(future)

        combined, _ = self.vsn_s(stafe_output)
        c_s = self.grn_s(combined.unsqueeze(1))
        obs_feature = torch.concat([confe_output, catfe_output], axis=-2)
        
        vsn_future_output, _ = self.vsn_future(futfe_output, c_s)
        vsn_obs_output, _ = self.vsn(obs_feature, c_s)
        vsn_output = torch.concat([vsn_obs_output, vsn_future_output.unsqueeze(1)], axis=1)
        
        lstm_enc_output, (c_t, h_t) = self.lstm_encoder(vsn_obs_output, (c_s.transpose(1, 0), c_s.transpose(1, 0)))
        lstm_dec_output, _ = self.lstm_decoder(vsn_future_output.unsqueeze(1), (c_t, h_t))
        lstm_output = torch.concat([lstm_enc_output, lstm_dec_output], axis=1)
        
        gal1_output = self.gal1(lstm_output, vsn_output)
        static_enrich = self.static_grn(gal1_output, c_s)
        
        B, attn_weights = self.mha(query=static_enrich, key=static_enrich, value=static_enrich, attn_mask=self.temporal_mask)

        gal2_output = self.gal2(B[:, -self.tau:, :], B[:, -self.tau:, :])
        fgrn_output = self.final_grn(gal2_output)
        gal3_output = self.gal3(gal1_output[:, -self.tau:, :], fgrn_output)
        qo_output = self.quantile_layer(gal3_output)
        
        return qo_output, attn_weights
    
