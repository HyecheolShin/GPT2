"""
GPT 2.0 — WikiText-2 학습 스크립트
notebook_gpt2.ipynb의 코드를 직접 실행하고 결과를 저장합니다.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math
import matplotlib.pyplot as plt
import warnings
import json
import os

warnings.filterwarnings('ignore')
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'PyTorch: {torch.__version__}')
print()

# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────
print('==> WikiText-2 로딩 중...')
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')

train_text = '\n'.join(line for line in ds['train']['text']      if line.strip())
val_text   = '\n'.join(line for line in ds['validation']['text'] if line.strip())
test_text  = '\n'.join(line for line in ds['test']['text']       if line.strip())

print(f'Train: {len(train_text):>10,} chars')
print(f'Val:   {len(val_text):>10,} chars')
print(f'Test:  {len(test_text):>10,} chars')
print()

# ── 2. 어휘 구성 ────────────────────────────────────────────────────────────────
all_text = train_text + val_text + test_text
chars = sorted(set(all_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f'Vocabulary size: {vocab_size}')

train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
val_data   = torch.tensor([stoi[c] for c in val_text],   dtype=torch.long)
print(f'Train tokens: {len(train_data):,}')
print(f'Val tokens:   {len(val_data):,}')
print()

# ── 3. Dataset ─────────────────────────────────────────────────────────────────
class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

BLOCK_SIZE = 128
BATCH_SIZE = 128

train_dataset = NextTokenDataset(train_data, BLOCK_SIZE)
val_dataset   = NextTokenDataset(val_data,   BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ── 4. 모델 아키텍처 ────────────────────────────────────────────────────────────
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))

class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads   = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj    = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(emb_dim)
        self.sa   = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2  = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size,
                 emb_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f    = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape
        pos     = torch.arange(T, device=x.device)
        tok     = self.token_embedding(x)
        pos_emb = self.position_embedding(pos).unsqueeze(0)
        h = self.drop(tok + pos_emb)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.lm_head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = GPT2(vocab_size, BLOCK_SIZE).to(device)
print(f'GPT 2.0 파라미터 수: {model.num_params():,}')
print()

# ── 5. 학습 ────────────────────────────────────────────────────────────────────
def get_lr(step, max_lr=3e-4, warmup_steps=300, total_steps=5000):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, loader, device, max_batches=80):
    model.eval()
    total, count = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(xb.to(device))
        total += F.cross_entropy(logits.transpose(1, 2), yb.to(device)).item()
        count += 1
    model.train()
    return total / count

MAX_STEPS           = 5000
LOG_INTERVAL        = 100
CHECKPOINT_INTERVAL = 500
CHECKPOINT_PATH     = 'checkpoint.pt'

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
)

train_losses, val_losses, step_log = [], [], []
start_step = 0

# ── 체크포인트 복원 (있으면 이어서 학습) ─────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    print(f'==> checkpoint found: {CHECKPOINT_PATH} -> resuming training')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step   = ckpt['step'] + 1
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    step_log     = ckpt['step_log']
    print(f'    step {ckpt["step"]}부터 재개  (train {ckpt["last_train_loss"]:.4f} | val {ckpt["last_val_loss"]:.4f})')
else:
    print('==> 처음부터 학습 시작')

train_iter = iter(train_loader)
model.train()

for step in range(start_step, MAX_STEPS + 1):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr

    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    loss   = F.cross_entropy(logits.transpose(1, 2), yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        step_log.append(step)
        print(f'step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | lr {lr:.2e}')

    # ── 체크포인트 저장 ─────────────────────────────────────────────────────────
    if step % CHECKPOINT_INTERVAL == 0 and step > 0:
        torch.save({
            'step':               step,
            'model_state_dict':   model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses':       train_losses,
            'val_losses':         val_losses,
            'step_log':           step_log,
            'last_train_loss':    train_losses[-1],
            'last_val_loss':      val_losses[-1],
        }, CHECKPOINT_PATH)
        print(f'    [checkpoint saved -> step {step}]')

print('\n학습 완료!')
print(f'최종 Train Loss: {train_losses[-1]:.4f}')
print(f'최종 Val Loss:   {val_losses[-1]:.4f}')

# ── 6. 학습 곡선 저장 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(step_log, train_losses, label='Train Loss', color='steelblue', linewidth=2)
axes[0].plot(step_log, val_losses,   label='Val Loss',   color='tomato',    linewidth=2)
axes[0].set_xlabel('Step', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('GPT 2.0 Training on WikiText-2', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

lr_steps = list(range(MAX_STEPS + 1))
lr_vals  = [get_lr(s) for s in lr_steps]
axes[1].plot(lr_steps, lr_vals, color='darkorchid', linewidth=2)
axes[1].set_xlabel('Step', fontsize=12)
axes[1].set_ylabel('Learning Rate', fontsize=12)
axes[1].set_title('Cosine Annealing with Warmup', fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
print('학습 곡선 저장: training_curve.png')

# ── 7. 텍스트 생성 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, idx, max_new_tokens=400, temperature=1.0, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits   = model(idx_cond)[:, -1, :] / temperature
        if top_k is not None:
            threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits[logits < threshold] = float('-inf')
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx      = torch.cat([idx, next_tok], dim=1)
    model.train()
    return idx

def encode(text):
    return torch.tensor([[stoi.get(c, 0) for c in text]], dtype=torch.long, device=device)

def decode(tensor):
    return ''.join(itos[i.item()] for i in tensor[0])

print('\n==> 텍스트 생성')
generated_samples = {}

prompts_temps = [
    ('The history of ', 0.7),
    ('The history of ', 1.0),
    ('The history of ', 1.3),
    ('In the field of ', 0.8),
    ('According to the report , ', 0.8),
    ('The researchers found that ', 0.8),
]

for prompt, temp in prompts_temps:
    key = f'{prompt.strip()} (temp={temp})'
    ctx = encode(prompt)
    gen = generate(model, ctx, max_new_tokens=300, temperature=temp, top_k=40)
    text = decode(gen)
    generated_samples[key] = text
    print(f'\n{"=" * 60}')
    print(f'프롬프트: "{prompt}"  |  Temperature={temp}')
    print('=' * 60)
    print(text)

# 결과를 JSON으로 저장
results = {
    'final_train_loss': train_losses[-1],
    'final_val_loss':   val_losses[-1],
    'step_log':         step_log,
    'train_losses':     train_losses,
    'val_losses':       val_losses,
    'generated_samples': generated_samples,
}
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size':       vocab_size,
    'block_size':       BLOCK_SIZE,
    'stoi':             stoi,
    'itos':             {str(k): v for k, v in itos.items()},
    'final_train_loss': train_losses[-1],
    'final_val_loss':   val_losses[-1],
}, 'gpt2_wikitext2.pt')

print('\n모델 저장 완료: gpt2_wikitext2.pt')
print('결과 저장 완료: results.json')
print('학습 곡선 저장: training_curve.png')
