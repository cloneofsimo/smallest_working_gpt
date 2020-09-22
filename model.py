import math
import torch
import torch.nn as nn
from torch.nn import functional as F

emb_s = 64
head_cnt = 8
block_size = 128
n_layer = 8

dp1 = 0.1
dp2 = 0.1
dp3 = 0.1

emb = emb_s*head_cnt

class attention_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(emb, emb)
        self.q_proj = nn.Linear(emb, emb)
        self.v_proj = nn.Linear(emb, emb)

        self.dp = nn.Dropout(dp1)
        
        self.proj = nn.Linear(emb, emb)
        self.mask = torch.tril(torch.ones(block_size, block_size)).reshape(1, block_size, block_size).to("cuda:0")
        
        self.head_cnt = head_cnt
        self.emb_s = emb_s

        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.GELU(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp2),
        )

    def forward_single_attn(self, x):
        T = x.shape[1]

        k = self.k_proj(x) # (B, T, hs)
        q = self.q_proj(x)
        v = self.v_proj(x)

        att = q @ k.transpose(-2, -1) #att : (B, T, T)
        att = att * 1.0/math.sqrt(self.emb_s)
        att = att.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)

        y = att@v # B, T, hs
        return y
    
    def forward_multi_attn(self, x):
        splits = torch.split(x, self.emb_s, dim = -1)
        multihead_attn = torch.cat([self.forward_single_attn(tnsr) for tnsr in splits], dim = -1)
        self.dp(self.proj(multihead_attn))
    
    def forward(self, x):
        x = x + self.forward_single_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size = 2000):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, emb))
        self.drop = nn.Dropout(dp3)

        self.attns_blocks = nn.Sequential(*[attention_layer() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, vocab_size, bias = False)
        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
            return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        _, t = idx.size()

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.attns_blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
