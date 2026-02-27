import torch
import torch.nn as nn
from .drs import DirectionalResidualStack
from .qdt import QuestionGuidedDifferenceTokenizer, ClinicalBERTText
from .mrm import MaskedResidualModel
from .heads import TinyTransformerDecoder

class DiffVQAModel(nn.Module):
    def __init__(
        self,
        backbone="swin_tiny_patch4_window7_224",
        freeze_backbone=False,
        pretrained_weights_path=None,
        text_dim=768,
        text_proj_dim=768,
        topk=64,
        num_classes=8000,
        max_ans_len=100,
        mask_ratio=0.5,
        **kwargs
    ):
        super().__init__()
        
        # 1. Modern Backbone + CIDA
        self.drs = DirectionalResidualStack(
            backbone_name=backbone, 
            freeze_backbone=freeze_backbone,
            pretrained_weights_path=pretrained_weights_path
        )
        C = self.drs.ch # Get channel dim from Swin
        
        # 2. NOVELTY: Auxiliary Classification Head (for Multi-Task Learning)
        # Predicts 14 diseases
        self.classifier = nn.Sequential(
            nn.Linear(C, 256),
            nn.ReLU(),
            nn.Linear(256, 14) 
        )

        # 3. Text Encoder
        self.text = ClinicalBERTText(
            d_txt=text_dim, proj_dim=text_proj_dim, fine_tune=True
        )

        # 4. QDT with Semantic Guidance
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=C, d_txt=text_proj_dim, k=topk
        )

        # 5. MRM
        self.mrm = MaskedResidualModel(c_all=C, mask_ratio=mask_ratio)

        # 6. Decoder
        self.head = TinyTransformerDecoder(
            dim=C, vocab_size=num_classes, max_len=max_ans_len, 
            txt_dim=text_proj_dim
        )

    def forward(self, img_ref, img_cur, token_batch):
        # A. Visual Feature Extraction & CIDA
        out_drs = self.drs(img_ref, img_cur)
        f_diff = out_drs["diff_feat"]
        f_ref  = out_drs["f_ref"]
        f_cur  = out_drs["f_cur"]
        
        # B. Auxiliary Classification (Novelty 2)
        # Global Average Pool -> Classify
        logits_ref = self.classifier(f_ref.mean(dim=(2, 3)))
        logits_cur = self.classifier(f_cur.mean(dim=(2, 3)))
        
        # C. Text Encoding
        q_vec = self.text(token_batch)
        
        # D. QDT with Semantic Guidance (Novelty 3)
        # We use the current image's predicted labels to guide the tokenizer
        # "Look for regions related to the diseases we think are present"
        sel_tokens, heatmap, gate_l1 = self.qdt(
            q_vec, f_diff, semantic_prior=logits_cur
        )
        
        # E. MRM (Self-Supervised)
        mrm_out = self.mrm(f_diff)
        
        return {
            "sel_tokens": sel_tokens,
            "q_vec": q_vec,
            "logits_ref": logits_ref,
            "logits_cur": logits_cur,
            "heatmap": heatmap,
            "gate_l1": gate_l1,
            **mrm_out
        }