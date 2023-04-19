import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

class GPCopula(nn.Module):
    def __init__(self, d_input, d_hidden, tau, beta, rank, n_layers, dr, device, norm="standard"):
        super(GPCopula, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden 
        self.tau = tau
        self.device = device
        self.norm = norm 
        self.lstm_list = nn.ModuleList([nn.LSTM(1, d_hidden, n_layers, dropout=dr, batch_first=True) for _ in range(self.d_input)])
        
        self.d_map = nn.Linear(d_hidden, 1, bias=False)
        
        if self.norm == "min-max":
            self.mu_activation = nn.Sigmoid()
        else:
            self.mu_activation = nn.Identity()
        self.v_map = nn.Linear(d_hidden, rank, bias=False)
        self.mu_map = nn.Linear(d_hidden, 1, bias=False)
        self.softplus = nn.Softplus(beta=beta)
         
    def forward(self, x):
        hidden_list = []
        
        for i in range(self.d_input):
            _, (hidden, cell) = self.lstm_list[i](x[..., i:i+1])
            hidden_list.append(hidden[-1:, ...])

        _hidden = torch.cat(hidden_list, axis=0)
        _hidden = _hidden.transpose(1, 0)
            
        mu = self.mu_activation(self.mu_map(_hidden))

        d = self.softplus(self.d_map(_hidden))
        v = self.v_map(_hidden)
        
        sigma_term1 = torch.diag_embed(d.squeeze(-1))
        sigma_term2 = torch.matmul(v, torch.transpose(v, 2, 1))
        sigma = sigma_term1 + sigma_term2
        
        return mu, sigma

class GPNegL(nn.Module):
    def __init__(self, ecdf_list, device):
        super(GPNegL, self).__init__()
        self.ecdf_list = ecdf_list 
        self.device = device
        self.norm_dist = Normal(0, 1)
        import math
        self.pi = torch.tensor(math.pi).float().to(device)
        
    def forward(self, true, params):
        mu, sigma = params
        L, info = torch.linalg.cholesky_ex(sigma)
        L_inverse = torch.inverse(L)
        det_L = torch.det(L)
        
        emp_quantile_ = []
        for i in range(d_input):
            emp_quantile_.append(torch.tensor(self.ecdf_list[i](true[..., i:i+1])).float())
        emp_quantile = torch.cat(emp_quantile_, dim=-1).to(self.device)
        gc_output = self.norm_dist.icdf(torch.clip(emp_quantile, min=0.001, max=0.999))
        
        return torch.log(det_L).mean() + (0.5 *  torch.square(L_inverse @ (gc_output.transpose(1, 2) - mu))).sum(dim=1).mean()

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    total_loss = []
    
    for batch in loader:
        batch_input, batch_infer = batch 
        
        batch_input = batch_input.to(device)
        batch_infer = batch_infer.to(device)

        pred = model(batch_input)
        
        loss = criterion(batch_infer, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)
    
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        batch_input, batch_infer = batch 
        
        batch_input = batch_input.to(device)
        batch_infer = batch_infer.to(device)
        
        pred = model(batch_input)
        
        loss = criterion(batch_infer, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss) 