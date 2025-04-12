import math
import torch
from torch import nn

class LoRAAttention(nn.Module):
    def __init__(self, orig_attention, r=32, alpha=1.0):
        super().__init__()
        self.orig_attention = orig_attention
        self.r = r
        self.alpha = alpha
        self.query = orig_attention.q_proj
        self.key = orig_attention.k_proj
        self.value = orig_attention.v_proj

        self.A_q = nn.Parameter(torch.randn(r, self.query.weight.size(1)) * 0.01)
        self.B_q = nn.Parameter(torch.randn(self.query.weight.size(0), r) * 0.01)
        self.A_k = nn.Parameter(torch.randn(r, self.key.weight.size(1)) * 0.01)
        self.B_k = nn.Parameter(torch.randn(self.key.weight.size(0), r) * 0.01)
        self.A_v = nn.Parameter(torch.randn(r, self.value.weight.size(1)) * 0.01)
        self.B_v = nn.Parameter(torch.randn(self.value.weight.size(0), r) * 0.01)

        self.query.weight.requires_grad = False
        self.key.weight.requires_grad = False
        self.value.weight.requires_grad = False

    def forward(self, hidden_states, attention_mask=None):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        lora_q = (hidden_states @ self.A_q.T) @ self.B_q.T
        lora_k = (hidden_states @ self.A_k.T) @ self.B_k.T
        lora_v = (hidden_states @ self.A_v.T) @ self.B_v.T

        q = q + self.alpha * lora_q
        k = k + self.alpha * lora_k
        v = v + self.alpha * lora_v

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, v)

        return output

def lora_adapter(module, device):
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, LoRAAttention(child).to(device))
        else:
            lora_adapter(child, device)
