import modal
import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import math
import os
import numpy as np
import time
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 1. Define Modal App and Container Image with PyTorch dependencies
app = modal.App("nano-gpt-training")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
    .pip_install("tokenizers")
    .pip_install("numpy")
    .add_local_file("dataset.txt", remote_path="/root/dataset.txt")
    .add_local_file("tokenizer.json", remote_path="/root/tokenizer.json")
)

# 2. Use a Modal Volume to persist your dataset and model checkpoints
volume = modal.Volume.from_name("gpt-data-vol", create_if_missing=True)
VOLUME_DIR = "/root/data"

def get_memory_usage():
    device_id = torch.cuda.current_device()
        
    # Get allocated memory (tensors currently in use by your model/data)
    allocated_vram = torch.cuda.memory_allocated(device_id) / (1024 ** 3) # Converted to GB
    
    # Get reserved memory (total memory managed by PyTorch's caching allocator)
    reserved_vram = torch.cuda.memory_reserved(device_id) / (1024 ** 3) # Converted to GB
    
    # Get total available VRAM on your GPU card
    total_vram = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3) # Converted to GB
    
    return allocated_vram, reserved_vram, total_vram
    

@app.function(image=image, gpu="H100", volumes={VOLUME_DIR: volume}, timeout=7200)
def train_model():    
    # Training parameters
    batch_size = 128
    block_size = 512

    learning_rate = 3e-4
    warmup_iters = 500
    min_lr = 3e-5

    dropout = 0.1

    # Model parameters
    n_embd = 768
    n_head = 12
    n_layer = 12

    # Training budget
    max_iters = 25000

    # Evaluation
    eval_interval = 500
    eval_iters = 30

    # Hardware
    device = 'cuda'
    ptdtype = torch.bfloat16

    #Prepearing the dataset
    print("preparing the dataset...")
    with open('dataset.txt', 'r') as f:
        text = f.read()
        
    print("length of data:", len(text), "characters")
    
    # Load the tokenizer from the saved file
    tokenizer = Tokenizer.from_file("tokenizer.json")

    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")
    print("")


    print("Starting chunked tokenization...", flush=True)

    chunk_size = 100_000
    batch_chunks = 64
    # Set before using the tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Tune this to the CPU allocation of your Modal container
    #os.environ["RAYON_NUM_THREADS"] = "64"

    # Divide text into manageable pieces
    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    token_ids = []
    start = time.time()

    for i in range(0, len(chunks), batch_chunks):

        batch = chunks[i:i + batch_chunks]

        encodings = tokenizer.encode_batch(
            batch,
            add_special_tokens=False
        )

        for encoding in encodings:
            token_ids.extend(encoding.ids)

        if i % (batch_chunks * 10) == 0:
            progress = min(i + batch_chunks, len(chunks)) / len(chunks) * 100

            print(
                f"Tokenized {progress:.1f}% "
                f"| Tokens: {len(token_ids):,} "
                f"| Time: {time.time() - start:.1f}s",
                flush=True
            )

    print("Tokenization complete!", flush=True)

    data = torch.tensor(token_ids, dtype=torch.long)

    print("Total tokens:", len(data))

    data = torch.tensor(token_ids, dtype=torch.long)

    print("Total tokens:", len(data))

    #Splitting the dataset into train and validation
    print("Splitting the dataset into train and validation...")
    n = int(0.9 * len(data))
    train_data = data[:n]
    print("------------------------")
    print("train data length:", len(train_data), "tokens")
    print("Total epochs:", (batch_size * block_size * max_iters) / len(train_data))
    print("------------------------")
    val_data = data[n:]
    
    def get_lr(it):
        # 1. Linear warmup phase
        if it < warmup_iters:
            return learning_rate * (it + 1) / warmup_iters
        
        # 2. If past max_iters, stay at minimum learning rate
        if it > max_iters:
            return min_lr
            
        # 3. Cosine decay phase down to min_lr
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Goes from 1.0 down to 0.0
        return min_lr + coeff * (learning_rate - min_lr)

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
                with torch.amp.autocast(device_type=device, dtype=ptdtype):
                    logits, loss = model(X, Y)  
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    xb, yb = get_batch('train')

    #Training the model
    torch.manual_seed(1337)

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            # Key, query, value projections for all heads combined in a single linear layer
            self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
            # Output projection layer
            self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
            # Regularization / Dropout
            self.attn_dropout = nn.Dropout(dropout)
            self.resid_dropout = nn.Dropout(dropout)
            
            # Check if PyTorch supports scaled_dot_product_attention (PyTorch >= 2.0)
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        def forward(self, x):
            B, T, C = x.shape  # Batch size, sequence length, embedding dim (n_embd)
            
            # 1. Calculate query, key, and values for all heads simultaneously
            q, k, v = self.c_attn(x).split(n_embd, dim=2)
            
            # 2. Reshape and transpose to separate the heads: (B, T, n_head, head_size) -> (B, n_head, T, head_size)
            head_size = C // n_head
            k = k.view(B, T, n_head, head_size).transpose(1, 2)
            q = q.view(B, T, n_head, head_size).transpose(1, 2)
            v = v.view(B, T, n_head, head_size).transpose(1, 2)

            # 3. Apply Fast FlashAttention / Scaled Dot Product Attention
            if self.flash:
                # PyTorch's optimized hardware kernel handles scaling, masking, and softmax natively
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=dropout if self.training else 0.0, 
                    is_causal=True
                )
            else:
                # Fallback manual calculation if using an older PyTorch version
                wei = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)
                # Create a dynamic causal mask on the fly if needed
                tril = torch.tril(torch.ones(T, T, device=x.device))
                wei = wei.masked_fill(tril == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.attn_dropout(wei)
                y = wei @ v

            # 4. Re-assemble all head outputs side-by-side and project back to embedding space
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            output = self.resid_dropout(self.c_proj(y))
            return output
        
    class FeedForward(nn.Module):
        """ a simple linear layer followed by a non-linearity """

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
        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = CausalSelfAttention()
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
        
        @torch.no_grad()
        def generate(self, idx, max_new_tokens):
            was_training = self.training
            self.eval()

            try:
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -block_size:]

                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :]

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    idx = torch.cat((idx, idx_next), dim=1)

            finally:
                self.train(was_training)

            return idx

    base_model = GPTLanguageModel().to(device)
    model = torch.compile(base_model)

    with torch.amp.autocast(device_type=device, dtype=ptdtype):
        logits, loss = model(xb, yb)
    print("logits shape:", logits.shape)
    print("loss:", loss)
    print("Output:", tokenizer.decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
    print("")

    #Training the model
    print("Training the model...")
    start_time = datetime.now().timestamp()
    step_start_time = datetime.now().timestamp()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("")

    for step in range(max_iters):
        # Get the adjusted learning rate for this specific step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if step % eval_interval == 0:
            losses = estimate_loss()
            vram_allocated, vram_reserved, vram_total = get_memory_usage()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time taken: {datetime.now().timestamp() - step_start_time:.2f} seconds, VRAM allocated: {vram_allocated:.2f} GB, VRAM reserved: {vram_reserved:.2f} GB, Total VRAM: {vram_total:.2f} GB, learning rate: {lr:.6f}")
            step_start_time = datetime.now().timestamp()
        
        if step % (eval_interval*5) == 0:
            # Save model checkpoint every 5 evaluation intervals
            checkpoint = {
                'step': step,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparameters': {
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                    'vocab_size': vocab_size,
                    'dropout': dropout
                }
            }  
            file_name = f"my_custom_gpt_step_{step}_train_loss_{loss.item():.4f}_val_loss_{losses['val']:.4f}.pt"
            checkpoint_path = os.path.join(VOLUME_DIR, file_name)
            torch.save(checkpoint, checkpoint_path)
            volume.commit()
            #Sample output from the model after saving checkpoint
            prompt = "#TITLE#Cena piva roste!\n#CONTENT#"
            context = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device).unsqueeze(0)
            output = tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist(), skip_special_tokens=False)
            print(f"Sample output: {output}")

            
        #Get batch
        xb, yb = get_batch('train')

        #Forward pass
        with torch.amp.autocast(device_type=device, dtype=ptdtype):
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
    print("Output:", tokenizer.decode(model.generate(context, max_new_tokens=300)[0].tolist(), skip_special_tokens=False))
    end_time = datetime.now().timestamp()
    print("Time taken to generate text:", end_time - start_time, "seconds")
    print("")
    
    checkpoint = {
        'step': step,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'vocab_size': vocab_size,
            'dropout': dropout
        }
    }

    checkpoint_path = os.path.join(VOLUME_DIR, "my_custom_gpt.pt")
    torch.save(checkpoint, checkpoint_path)
    volume.commit()
    print("Model saved successfully!")
    
@app.local_entrypoint()
def main():
    # This runs locally when you type `modal run modal_train.py`
    print("Triggering remote training run on Modal...")
    train_model.remote()