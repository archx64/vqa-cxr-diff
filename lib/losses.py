import torch
import torch.nn.functional as F


def heatmap_kl(h1, h2, eps=1e-6):
    # KL(P||Q) + KL(Q||P) symmetric
    p = h1.float() + eps
    q = h2.float() + eps
    p = p / p.sum(dim=(1, 2), keepdim=True)
    q = q / q.sum(dim=(1, 2), keepdim=True)
    kl1 = (p * (p.log() - q.log())).sum(dim=(1, 2))
    kl2 = (q * (q.log() - p.log())).sum(dim=(1, 2))
    return (kl1 + kl2).mean()


def info_nce_token_sets(toks_a, toks_b, temperature=0.07):
    """
    toks_*: (B,k,D) — mean-pool and contrast batch-wise.
    """
    a = toks_a.mean(dim=1)  # (B,D)
    b = toks_b.mean(dim=1)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)
