import torch
import torch.nn as nn
import torch.nn.functional as F


class rnn(nn.Module):
    def __init__(self, emb = 64, vocab_size = 10, block_size = 10):
        super().__init__()
        self.emb = emb
        self.beg_fc = nn.Linear(emb, 2*emb) #used only once
        self.mlp = nn.Sequential(
            nn.Linear(2*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 8*emb),
            nn.ReLU(),
            nn.Linear(8*emb, 8*emb),
            nn.ReLU(),
            nn.Linear(8*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 2*emb),
            nn.ReLU(),
        )# 2*emb -> 2*emb, top is result, buttom is recurrent..
        
        self.ln1 = nn.LayerNorm(2*emb)

        self.head = nn.Sequential(
            nn.Linear(emb, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size),
        )
        self.tok_emb = nn.Embedding(vocab_size, emb)
        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
            return self.block_size  

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)  
    
    def forward(self, x, targets = None):
        x = self.tok_emb(x)
        B, T, C = x.size()
        x_s = x[:,0,:]
        result = torch.zeros(B, T, C).to("cuda:0")
        x_s = self.beg_fc(x_s)
        result[:,0,:] = x_s[:,self.emb:]

        for idx in range(1, T):
    
            x_s = torch.cat([x[:, idx, :], x_s[:, :self.emb]], dim = -1)
            x_s = self.mlp(x_s)
            x_s = self.ln1(x_s)
            result[:,idx,:] = x_s[:,self.emb:]

        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


class lstm(nn.Module):
    def __init__(self, emb = 64, vocab_size = 10, block_size = 10):
        super().__init__()
        self.emb = emb

        self.beg_lt = nn.Linear(emb, emb)
        self.beg_st = nn.Linear(emb, emb)

        self.mlp_st_lt_mul = nn.Sequential(
            nn.Linear(2*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 8*emb),
            nn.ReLU(),
            nn.Linear(8*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 2*emb),
            nn.ReLU(),
            nn.Linear(2*emb, emb),
            nn.Sigmoid(),
        )# 2*emb -> emb
        self.mlp_st_lt_sig = nn.Sequential(
            nn.Linear(2*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 2*emb),
            nn.ReLU(),
            nn.Linear(2*emb, emb),
            nn.Sigmoid(),
        )
        self.mlp_st_lt_tanh = nn.Sequential(
            nn.Linear(2*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 2*emb),
            nn.ReLU(),
            nn.Linear(2*emb, emb),
            nn.Tanh(),
        )
        self.mlp_lt_st = nn.Sequential(
            nn.Linear(2*emb, 4*emb),
            nn.ReLU(),
            nn.Linear(4*emb, 2*emb),
            nn.ReLU(),
            nn.Linear(2*emb, emb),
            nn.Sigmoid(),
        )

        self.ln1 = nn.LayerNorm(emb)
        self.head = nn.Sequential(
            nn.Linear(emb, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size),
        )
        
        self.tok_emb = nn.Embedding(vocab_size, emb)
        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
            return self.block_size 

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, targets = None):
        x = self.tok_emb(x)
        B, T, C = x.size()
        x_s = x[:,0,:]
        result = torch.zeros(B, T, C).to("cuda:0")
        x_st = self.beg_st(x_s)
        x_lt = self.beg_lt(x_s)
        x_lt = self.ln1(x_lt)
        x_st = self.ln1(x_st)
        result[:,0,:] = self.ln1(x_st)

        for idx in range(1, T):
    
            x_st = torch.cat([x[:, idx, :], x_s], dim = -1)
            x_lt = self.mlp_st_lt_mul(x_st)*x_lt
            x_lt = x_lt + self.mlp_st_lt_sig(x_st)*self.mlp_st_lt_tanh(x_st)
            x_st = torch.tanh(x_lt)*self.mlp_lt_st(x_st)
            
            x_lt = self.ln1(x_lt)
            x_st = self.ln1(x_st)
            
            result[:,idx,:] = x_st

        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss