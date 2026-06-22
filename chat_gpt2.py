import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# ── 모델 정의 (train_gpt2.py와 동일) ──────────────────────────────────────────
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
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj  = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), GELU(),
            nn.Linear(4 * emb_dim, emb_dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

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
    def __init__(self, vocab_size, block_size, emb_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f    = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.token_embedding(x) + self.position_embedding(torch.arange(T, device=x.device)))
        return self.lm_head(self.ln_f(self.blocks(h)))

# ── 모델 로딩 ──────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'gpt2_wikitext2.pt'

if not os.path.exists(MODEL_PATH):
    print(f'모델 파일을 찾을 수 없습니다: {MODEL_PATH}')
    exit(1)

print('GPT 2.0 모델 로딩 중...')
ckpt = torch.load(MODEL_PATH, map_location=device)

stoi = ckpt['stoi']
itos = {int(k): v for k, v in ckpt['itos'].items()}

model = GPT2(ckpt['vocab_size'], ckpt['block_size']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'로딩 완료! (device: {device})')
print()

# ── 생성 함수 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size:])[:, -1, :] / temperature
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits[logits < threshold] = float('-inf')
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, next_tok], dim=1)
    return ''.join(itos[i.item()] for i in idx[0])

# ── 인터랙티브 루프 ────────────────────────────────────────────────────────────
print('=' * 60)
print('GPT 2.0 텍스트 생성기 (WikiText-2 학습)')
print('프롬프트를 입력하면 이어지는 텍스트를 생성합니다.')
print('종료: Ctrl+C 또는 "quit" 입력')
print('=' * 60)
print()

while True:
    try:
        prompt = input('프롬프트> ').strip()
        if not prompt or prompt.lower() == 'quit':
            break

        print()
        print(generate(prompt))
        print()
    except KeyboardInterrupt:
        break

print('종료합니다.')
