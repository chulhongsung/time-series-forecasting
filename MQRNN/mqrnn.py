import torch
import torch.nn as nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, d_input:int, d_embedding:int, n_embedding:list, d_model:int, n_layers:int, dr:float):
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

        return hidden, cell # (num_layers, batch_size, d_model)
    
class GlobalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_embedding:int, n_embedding:list, d_model:int, tau:int, num_targets:int, dr:float):
        super(GlobalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.n_embedding = n_embedding
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.linear_layers = nn.ModuleList([nn.Linear(d_hidden + tau * d_embedding * len(n_embedding), (tau+1) * d_model) for _ in range(num_targets)])
        self.dropout = nn.Dropout(dr)
        
    def forward(self, future, hidden):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(future[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        emb_output_ = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output_.view(future.size(0), -1)
        
        num_layers, batch_size, d_hidden = hidden.size()
        
        assert d_hidden == self.d_model 
        
        x = torch.cat([hidden[num_layers-1], emb_output], axis=-1)
        
        tmp_global_context = []
        for l in self.linear_layers:
            tmp_gc = self.dropout(l(x))
            tmp_global_context.append(tmp_gc.unsqueeze(1))
        
        global_context = torch.cat(tmp_global_context, axis=1)
        
        return emb_output_.view(batch_size, self.tau, -1), global_context # (batch_size, tau, d_embedding * len(n_embedding)), (batch_size, num_targets, (tau+1) * d_model), (tau+1): c_{a} , c_{t+1:t+tau}
    
class LocalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_embedding:int, n_embedding: list, d_model:int, tau:int, num_targets:int, num_quantiles:int, dr:float):
        super(LocalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.n_embedding = n_embedding
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = nn.Sequential(
            nn.Linear(2 * d_model + d_embedding * len(n_embedding), d_model * 2),
            nn.Dropout(dr),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, num_quantiles)            
            )
                
    def forward(self, embedded_future, global_output):
        batch_size = global_output.size(0)
        
        c_a = global_output[..., :self.d_model].unsqueeze(-2).repeat(1, 1, self.tau, 1) # (batch_size, num_targets, tau, d_model)
        c_t = global_output[..., self.d_model:].view(batch_size, self.num_targets, self.tau, -1) # (batch_size, num_targets, tau, d_model)
        x_ = torch.cat([c_a,c_t.view(batch_size, self.num_targets, self.tau, -1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model)
        x = torch.cat([x_, embedded_future.unsqueeze(1).repeat(1, self.num_targets, 1, 1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model + d_embedding * len(n_embedding))
        
        output = self.linear_layers(x)
        
        return output # (batch_size, num_targets, tau, num_quantiles)

class MQRNN(nn.Module):
    def __init__(self, d_input:int, d_embedding:int, n_embedding:list, d_model:int, tau:int, num_targets:int, num_quantiles: int, n_layers:int, dr:float):
        super(MQRNN, self).__init__()
        self.encoder = Encoder(
                               d_input=d_input,
                               d_embedding=d_embedding,
                               n_embedding=n_embedding,
                               d_model=d_model,
                               n_layers=n_layers,
                               dr=dr
                               )
        self.global_decoder = GlobalDecoder(
                                            d_hidden=d_model,
                                            d_embedding=d_embedding,
                                            n_embedding=n_embedding,
                                            d_model=d_model,
                                            tau=tau,
                                            num_targets=num_targets,
                                            dr=dr
                                            )
        self.local_decoder = LocalDecoder(
                                          d_hidden=d_model,
                                          d_embedding=d_embedding,
                                          n_embedding=n_embedding,
                                          d_model=d_model,
                                          tau=tau,
                                          num_targets=num_targets,
                                          num_quantiles=num_quantiles,
                                          dr=dr
                                          )
        
    def forward(self, conti, cate, future):
        hidden, _ = self.encoder(conti, cate)
        embedded_future, global_output = self.global_decoder(future, hidden)
        output = self.local_decoder(embedded_future, global_output)
        
        return output