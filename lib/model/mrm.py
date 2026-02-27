import torch
from torch import nn
from torch.nn import functional as F

class MaskedResidualModel(nn.Module):
    def __init__(self, c_all, mask_ratio=0.5):
        """
        Learns to reconstruct the 'Difference Features' (f_diff) from masked inputs.
        This forces the model to understand the context of the difference.
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Pre-processing adapter
        self.pre = nn.Sequential(
            nn.Conv2d(c_all, c_all, 1),
            nn.GELU(),
            nn.Conv2d(c_all, c_all, 1),
        )
        
        # Transformer-style projection
        self.enc = nn.Linear(c_all, c_all)
        self.dec = nn.Linear(c_all, c_all)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, c_all))

    def forward(self, feats):  
        """
        Args:
            feats: (B, C, H, W) - The output from the Difference Module (CIDA)
        """
        B, C, H, W = feats.shape

        # The target we want to reconstruct is the raw difference feature map
        target_patches = feats.flatten(2).transpose(1, 2) # (B, N, C) where N=H*W

        # Process input before masking
        x = self.pre(feats)
        input_patches = x.flatten(2).transpose(1, 2)  # (B, N, C)

        device = input_patches.device
        N = input_patches.shape[1]

        # --- Masking Logic ---
        num_mask = int(self.mask_ratio * N)
        
        # Generate random noise to sort and pick mask indices
        rand = torch.rand(B, N, device=device).argsort(-1)
        masked_idx = rand[:, :num_mask]
        unmasked_idx = rand[:, num_mask:]
        
        # Create batch index helper
        b_idx = torch.arange(B, device=device)[:, None]

        # Encoder only sees unmasked patches
        enc_all = self.enc(input_patches)
        enc_unmasked = enc_all[b_idx, unmasked_idx] # (B, N_unmasked, C)

        # --- Reconstruction ---
        # Create a full sequence filled with mask tokens
        full_seq = self.mask_token.expand(B, N, C).clone()
        
        # Fill in the unmasked parts with encoded features
        full_seq[b_idx, unmasked_idx] = enc_unmasked.to(full_seq.dtype)

        # Decoder tries to predict the whole sequence (or just masked parts)
        recon_all = self.dec(full_seq)
        
        # Extract predictions at masked locations
        recon_masked = recon_all[b_idx, masked_idx]

        # Extract targets at masked locations
        orig_masked_target = target_patches[b_idx, masked_idx]

        # Normalization for stability (LayerNorm over the channel dim)
        orig_masked_target = F.layer_norm(orig_masked_target, (C,))

        # Calculate MSE Loss
        loss = F.mse_loss(recon_masked, orig_masked_target)
        
        return {
            "loss_mrm": loss,
            # We return these for potential visualization or NCE losses
            "patches": target_patches, 
            "recon_all": recon_all,
        }