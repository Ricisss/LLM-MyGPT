import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 1. Device configuration (uses local GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. Load the checkpoint file
print("Loading checkpoint...")
checkpoint = torch.load('my_custom_gpt.pt', map_location=device)

# Extract saved configurations and dictionaries
model_state_dict = checkpoint['model_state_dict']
hp = checkpoint['hyperparameters']

n_embd = hp['n_embd']
n_head = hp['n_head']
n_layer = hp['n_layer']
block_size = hp['block_size']
vocab_size = hp['vocab_size']
dropout = hp['dropout']

# 3. Load the tokenizer from the saved file
tokenizer = Tokenizer.from_file("tokenizer.json")

# 3. Define the Model Architecture Classes (must match your training architecture)
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        head_size = C // n_head
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0.0, is_causal=True)
        else:
            wei = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)
            tril = torch.tril(torch.ones(T, T, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.attn_dropout(wei)
            y = wei @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

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
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 4. Initialize model instance and load trained weights
model = GPTLanguageModel().to(device)
model.load_state_dict(model_state_dict)
model.eval()
print("Model loaded successfully!")

# 5. Generate text
prompt = "#TITLE#"
context = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device).unsqueeze(0)

time_start = datetime.now()

generated_tokens = model.generate(context, max_new_tokens=256)[0].tolist()
print("\n--- Generated Output ---")
print(tokenizer.decode(generated_tokens, skip_special_tokens=False))
print("------------------------")
print(f"Time taken for generation: {(datetime.now() - time_start).total_seconds():.2f} seconds")
print(f"Tokens per second: {len(generated_tokens) / (datetime.now() - time_start).total_seconds():.2f}")