# multivariate version

import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, n_layers=3, dr=0.1):
        super(Encoder, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input + len(n_embedding) * d_embedding, d_model, n_layers, dropout=dr, batch_first=True)
        
    def forward(self, conti, cate):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(cate[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        emb_output = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output.view(conti.size(0), conti.size(1), -1)
        
        x = torch.cat([conti, emb_output], axis=-1)
        
        _, (hidden, cell) = self.lstm(x)

        return hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, num_targets, n_layers=3, dr=0.1):
        super(LSTMDecoder, self).__init__()
        self.n_layers = n_layers
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input + len(n_embedding) * d_embedding, d_model, n_layers, dropout=dr, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, num_targets)
        self.linear2 = nn.Linear(d_model, num_targets)
        self.dropout = nn.Dropout(dr)

    def forward(self, future, hidden, cell):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(future[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
        
        tau = future.size(1)    
        
        emb_output = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output.view(future.size(0), tau, -1) # (batch_size, tau, len(n_embedding) * d_embedding)
        
        lstm_output = []
   
        for t in range(tau):
            lstm_input = torch.cat([hidden[self.n_layers-1:self.n_layers].transpose(1, 0), emb_output[:, t:t+1, :]], axis=-1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
            lstm_output.append(output)
        
        lstm_output = torch.cat(lstm_output, axis=1)
        
        mu = self.linear1(lstm_output)
        sigma = torch.log(1+ torch.exp(self.linear2(lstm_output)))
        
        return mu, sigma

class DeepAR(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, num_targets, n_layers=3, dr=0.1):
        super(DeepAR, self).__init__()

        self.encoder = Encoder(d_input, d_embedding, n_embedding, d_model, n_layers, dr)
        self.decoder = LSTMDecoder(d_model, d_embedding, n_embedding, d_model, num_targets, n_layers, dr)

    def forward(self, conti, cate, future):
        
        encoder_hidden, encoder_cell = self.encoder(conti, cate)
        mu, sigma = self.decoder(future, encoder_hidden, encoder_cell)
        
        return mu, sigma
    
class NegativeGaussianLogLikelihood(nn.Module):
    def __init__(self, device):
        super(NegativeGaussianLogLikelihood, self).__init__()
        self.pi = torch.tensor(math.pi).float().to(device)
        
    def forward(self, true, mu, sigma):
        return (torch.square(true - mu)/(2*sigma) + torch.log(2*self.pi)/2).sum()
