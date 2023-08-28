import torch
import torch.nn as nn

class CateFeatureEmbedding(nn.Module):
    def __init__(self, d_embedding, n_embedding):
        super(CateFeatureEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        
    def forward(self, x):
        
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(x[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        return torch.cat(tmp_feature_list, axis=-2)

class ContiFeatureEmbedding(nn.Module):
    def __init__(self, d_embedding, num_rv):
        super(ContiFeatureEmbedding, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1, d_embedding) for _ in range(num_rv)])
        
    def forward(self, x):
        tmp_feature_list = []
        for i, l in enumerate(self.linears):
            tmp_feature = l(x[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        return torch.stack(tmp_feature_list, axis=-1).transpose(-1, -2)
    
class GLU(nn.Module):
    def __init__(self, d_model):
        super(GLU, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.LazyLinear(d_model)
        
    def forward(self, x):
        return torch.mul(self.sigmoid(self.linear1(x)), self.linear2(x))

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, dr):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.dropout1 = nn.Dropout(dr)
        self.linear2 = nn.LazyLinear(d_model)
        self.linear3 = nn.LazyLinear(d_model)
        self.linear4 = nn.LazyLinear(d_model)
        self.dropout2 = nn.Dropout(dr)
        self.elu = nn.ELU()
        self.glu = GLU(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        
    def forward(self, x, c=None):
        if c == None:
            c = torch.zeros_like(x)
        eta_2 = self.dropout1(self.linear1(x) + self.linear2(c))
        eta_1 = self.elu(self.dropout2(self.linear3(eta_2)))
        grn_output_ = self.glu(eta_1)
        grn_output = self.layer_norm(self.linear4(x) + grn_output_)
        return grn_output

class GAL(nn.Module):
    def __init__(self, d_model):
        super(GAL, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LazyLinear(d_model)
        
    def forward(self, resid, x):
        return self.layer_norm(torch.mul(self.sigmoid(resid), resid) + x)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = nn.Softmax(dim=-1)
        self.xi_grn = nn.ModuleList([GatedResidualNetwork(d_model, dr) for _ in range(d_input)])
        
    def forward(self, xi, c=None):
        Xi = xi.reshape(xi.size(0), xi.size(1), -1)
        weights = self.softmax(self.v_grn(Xi, c)).unsqueeze(-1)
        
        tmp_xi_list = []
        for i, l in enumerate(self.xi_grn):
            tmp_xi = l(xi[:, :, i:i+1])
            tmp_xi_list.append(tmp_xi)
        xi_list = torch.cat(tmp_xi_list, axis=-2)
        
        combined = torch.matmul(weights.transpose(3, 2), xi_list).squeeze()
        
        return combined, weights

class QuantileOutput(nn.Module):
    def __init__(self, tau, quantile):
        super(QuantileOutput, self).__init__()
        self.tau = tau
        self.quantile_linears = nn.ModuleList([nn.LazyLinear(1) for _ in range(len(quantile))])
        
    def forward(self, varphi):
        total_output_list = []
        
        for _, l in enumerate(self.quantile_linears):
            tmp_quantile_list = []
            
            for t in range(self.tau-1):
                tmp_quantile = l(varphi[:, (-self.tau + t) : (-self.tau + t + 1), ...])
                tmp_quantile_list.append(tmp_quantile)
            
            tmp_quantile_list.append(l(varphi[:, -1:, ...]))
            
            total_output_list.append(torch.cat(tmp_quantile_list, dim=1))
            
        return torch.cat(total_output_list, dim=-1)