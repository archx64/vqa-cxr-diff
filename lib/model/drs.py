import torch
from torch import nn
from torch.nn import functional as F
import timm
import logging

logger = logging.getLogger(__name__)

class CrossImageDifferenceAttention(nn.Module):
    """
    NOVELTY COMPONENT 1:
    Instead of hard subtraction (Cur - Ref), we use Attention to model
    the 'evolution' of features. This handles slight misalignment and 
    highlights semantic changes.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, use_cida=True):
        super().__init__()
        self.dim = dim
        self.use_cida = use_cida
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
        # Gating mechanism to weigh the difference
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, feat_ref, feat_cur):
        # inputs: (B, C, H, W) -> need (B, N, C) for Attention
        B, C, H, W = feat_cur.shape
        
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        cur_flat = feat_cur.flatten(2).transpose(1, 2)
        ref_flat = feat_ref.flatten(2).transpose(1, 2)

        if self.use_cida:
            # 1. Cross Attention: Query=Cur, Key=Ref, Val=Ref
            # "Reconstruct Current Image using patches from Reference Image"
            # If a patch in Cur is new (disease), it won't find a good match in Ref.
            aligned_ref, _ = self.cross_attn(query=cur_flat, key=ref_flat, value=ref_flat)

        else:
            # 1. By pass semantic alignment and assume perfect spatial correspondence
            aligned_ref = ref_flat

        
        # 2. Soft Difference
        # Features that exist in Cur but NOT in Aligned Ref are likely new pathologies
        diff = cur_flat - aligned_ref
        
        # 3. Gating (Technical Novelty: Learnable Subtraction)
        # Decide which differences are important
        gate_input = torch.cat([cur_flat, diff], dim=-1)
        importance = self.gate(gate_input)
        
        # weighted difference
        refined_diff = diff * importance
        refined_diff = self.out_proj(refined_diff)
        
        # Reshape back to spatial map for downstream modules
        # (B, H*W, C) -> (B, C, H, W)
        out = refined_diff.transpose(1, 2).reshape(B, C, H, W)
        
        return out, refined_diff

class DirectionalResidualStack(nn.Module):
    def __init__(self, backbone_name, freeze_backbone=False, pretrained_weights_path=None, use_cida=True):
        super().__init__()
        
        # Load Swin Transformer
        logger.info(f"Loading Backbone: {backbone_name}")
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True, out_indices=[-1]
        )
        
        # Custom Weights Loading (CXR-CLIP)
        if pretrained_weights_path:
            self._load_custom_weights(pretrained_weights_path)

        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        else:
            for p in self.backbone.parameters(): p.requires_grad = True

        # Get channel dimension
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)[0]
        #     feats = self.backbone(dummy)[0]
        #     # Swin in timm often outputs (B, H, W, C). We need to check.
        #     if feats.shape[1] != feats.shape[3]: # Likely (B, C, H, W) if it's standard features_only
        #         self.permute_needed = False
        #         self.ch = feats.shape[1]
        #     else:
        #         self.permute_needed = True # (B, H, W, C)
        #         self.ch = feats.shape[-1]

        # Robust channel detection: The channel dim is almost always larger than H/W
            if feats.shape[1] > feats.shape[-1]: 
                # Shape is (B, C, H, W) e.g., (1, 768, 7, 7)
                self.permute_needed = False
                self.ch = feats.shape[1] 
            else:
                # Shape is (B, H, W, C) e.g., (1, 7, 7, 768)
                self.permute_needed = True 
                self.ch = feats.shape[-1]



        logger.info(f"Backbone Channels: {self.ch}")

        # The Novel Difference Module
        self.cida = CrossImageDifferenceAttention(dim=self.ch, use_cida=use_cida)

    def _load_custom_weights(self, path):
        try:
            ckpt = torch.load(path, weights_only=False, map_location='cpu')
            if 'model' in ckpt: ckpt = ckpt['model']
            # Basic cleanup for common CLIP prefixes
            clean_ckpt = {k.replace("visual.", ""): v for k, v in ckpt.items() if "visual" in k or "backbone" in k}
            if not clean_ckpt: clean_ckpt = ckpt
            self.backbone.load_state_dict(clean_ckpt, strict=False)
            logger.info("Loaded custom backbone weights.")
        except Exception as e:
            logger.warning(f"Could not load custom weights: {e}")

    def extract_feat(self, img):
        # Extract and normalize shape to (B, C, H, W)
        f = self.backbone(img)[0]
        if self.permute_needed:
            f = f.permute(0, 3, 1, 2)
        return f

    def forward(self, img_ref, img_cur):
        f_ref = self.extract_feat(img_ref) # (B, C, H, W)
        f_cur = self.extract_feat(img_cur) # (B, C, H, W)
        
        # Use Novel Attention-based Difference
        diff_feat, _ = self.cida(f_ref, f_cur)
        
        return {
            "diff_feat": diff_feat, # The smart difference
            "f_ref": f_ref,         # Raw features for Aux Classification
            "f_cur": f_cur
        }