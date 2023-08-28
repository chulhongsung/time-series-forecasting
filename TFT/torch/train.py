import torch
import torch.nn as nn

class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(quantile).float().unsqueeze(0).unsqueeze(0).repeat(1, tau, 1).to(self.device)
    
    def forward(self, true, pred):
        true_rep = true.repeat(1, 1, len(self.quantile)).to(self.device)

        ql = torch.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr)*(pred - true_rep))
        
        return ql.mean()
    
def train(model, loader, criterion, optimizer, device):
    model.train()
    
    total_loss = []
    
    for batch in loader:
        conti_input, cate_input, static_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        static_input = static_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred, _ = model(conti_input, cate_input, static_input, future_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)

def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        conti_input, cate_input, static_input, future_input, true_y = batch
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        static_input = static_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred, _ = model(conti_input, cate_input, static_input, future_input)
        
        loss = criterion(true_y, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss) 