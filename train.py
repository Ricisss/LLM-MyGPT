import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size = 64
block_size = 256
max_iters = 3000
learning_rate = 1e-4
eval_interval = 300
device = 'cuda'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#Prepearing the dataset
print("preparing the dataset...")
with open('dataset.txt', 'r') as f:
    text = f.read()
    
print("length of data:", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print("vocab size:", vocab_size)
print("")

#Tokenization
print("Tokenization...")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encoder(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])
print("")

#Splitting the dataset into train and validation
print("Splitting the dataset into train and validation...")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data[:block_size+1])

torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')

#Training the model
torch.manual_seed(1337)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)        
            loss = F.cross_entropy(logits, targets)
        
        return logits,loss
    
    def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:] # crop context to block_size
                logits, loss = self(idx_cond) 
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)
                
                # Ensure idx_next is 2D just in case
                if idx_next.ndim == 1:
                    idx_next = idx_next.unsqueeze(1)
                    
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

model = GPTLanguageModel()
model = model.to(device)

logits, loss = model(xb, yb)
print("logits shape:", logits.shape)
print("loss:", loss)
print("Output:", decoder(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
print("")

#Training the model
print("Training the model...")
from datetime import datetime
start_time = datetime.now().timestamp()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print("")
print(f"Total parameters: {total_params:,}")
print("")

for steps in range(max_iters):
    
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #Get batch
    xb, yb = get_batch('train')

    #Forward pass
    logits, loss = model(xb, yb)
    
    #Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=device)
end_time = datetime.now().timestamp()
print("Time taken to train the model:", end_time - start_time, "seconds")
print("")

start_time = datetime.now().timestamp()
print("Output:", decoder(model.generate(context, max_new_tokens=300)[0].tolist()))
end_time = datetime.now().timestamp()
print("Time taken to generate text:", end_time - start_time, "seconds")
print("")