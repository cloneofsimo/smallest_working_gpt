  
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
import math
from model_rnn_lstm import rnn, lstm
from model_gpt import GPT

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
    
@torch.no_grad()
def sample(model, x, steps, temperature=1.0, top_k=12):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        
        logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x

class CharDataset(Dataset):
    def __init__(self, data, block_size, step_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-step_size], dtype=torch.long)
        y = torch.tensor(dix[step_size:], dtype=torch.long)
        return x, y


text = open('texts.txt', 'r', encoding='utf-8').read()
text = text[:10000]

epochs = 1
block_size = 64

batch_size = 128
lr = 1e-3
is_train = True
step_size = 1
train_dataset = CharDataset(text, block_size, step_size)
device = torch.device("cuda:0")
#model = lstm(emb = 32, vocab_size = train_dataset.vocab_size, block_size = block_size)
#model = rnn(emb = 32, vocab_size = train_dataset.vocab_size, block_size = block_size)
model = GPT(vocab_size = train_dataset.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-9)
dl = DataLoader(train_dataset, shuffle=True, pin_memory= True, batch_size=batch_size)

model.to(device)
model.train()
if is_train:
    for epoch in range(epochs):
        pbar = tqdm(dl)
        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch {epoch+1} train loss {loss.item():.5f}")
    torch.save(model.state_dict(), "model.dat")
else:
    model.load_state_dict(torch.load("model.dat"))

input_context = "여호와 "

x = torch.tensor([train_dataset.stoi[s] for s in input_context], dtype=torch.long)[None,...].to(device)
y = sample(model, x, 64, temperature=1.0, top_k= 12)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)