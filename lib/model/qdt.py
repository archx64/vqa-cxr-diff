import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

class ClinicalBERTText(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", d_txt=768,
                 proj_dim=256, fine_tune=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = d_txt
        self.proj = nn.Linear(self.out_dim, proj_dim) if proj_dim and proj_dim != self.out_dim else nn.Identity()

        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, questions, max_len=48):
        enc = self.tok(
            list(questions),
            padding=True,
            truncation=True, 
            max_length=max_len,
            return_tensors="pt"
        )
        return enc 

    def forward(self, token_batch):
        out = self.bert(**token_batch)
        cls = out.last_hidden_state[:, 0] 
        q = self.proj(cls)                
        return q

class QuestionGuidedDifferenceTokenizer(nn.Module):
    def __init__(self, c_img, d_txt, k=64, num_rows=3, num_cols=2):
        super().__init__()
        self.k = k
        
        # Projects visual features
        self.vis_proj = nn.Linear(c_img, c_img)
        
        # NOVELTY: Projects (Text + Predicted_Labels) -> Image Space
        # d_txt (768) + 14 (Labels) -> c_img
        self.sem_proj = nn.Linear(d_txt + 14, c_img) 
        
        self.scale = c_img ** -0.5
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, q_vec, feats, semantic_prior=None):
        """
        q_vec: (B, 768)
        feats: (B, C, H, W) - The 'diff_feat' from DRS
        semantic_prior: (B, 14) - The predicted logits from VQA model
        """
        B, C, H, W = feats.shape
        
        # 1. Flatten Visual Features
        vis_flat = feats.flatten(2).transpose(1, 2) # (B, N, C)
        vis_flat = self.vis_proj(vis_flat)
        
        # 2. Enhanced Query with Semantic Prior
        if semantic_prior is not None:
            # Concatenate Question Vector + Predicted Disease Logits
            q_enhanced = torch.cat([q_vec, semantic_prior], dim=1) # (B, 768+14)
        else:
            # Fallback (validation/inference without classification?)
            dummy_sem = torch.zeros(B, 14, device=q_vec.device)
            q_enhanced = torch.cat([q_vec, dummy_sem], dim=1)

        q_query = self.sem_proj(q_enhanced).unsqueeze(1) # (B, 1, C)
        
        # 3. Cross Attention (Dot Product)
        att = (q_query @ vis_flat.transpose(1, 2)) * self.scale
        att = att.transpose(1, 2) # (B, N, 1)
        
        # 4. Gating & Selection
        gated = self.gate(att) * att
        att_scores = gated.squeeze(-1)
        att_weights = F.softmax(att_scores, dim=-1)
        
        # Top-K Selection
        topk_vals, topk_idx = att_weights.topk(min(self.k, vis_flat.shape[1]), dim=-1)
        batch_idx = torch.arange(B, device=feats.device)[:, None]
        sel_tokens = vis_flat[batch_idx, topk_idx]
        
        return sel_tokens, att_weights.reshape(B, H, W), gated.abs().mean()