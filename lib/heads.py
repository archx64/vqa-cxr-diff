import torch
from torch import nn
from torch.nn import functional as F

class TinyTransformerDecoder(nn.Module):
    def __init__(self, dim=768, vocab_size=5000, nlayer=4, nhead=8, max_len=100, txt_dim=768, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))
        
        # increased depth and dropout for better regularization
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, 
            batch_first=True, dropout=dropout
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=nlayer)
        
        self.out = nn.Linear(dim, vocab_size)
        self.mem_proj = nn.Linear(dim, dim)
        
        # projects the question/semantic vector to the decoder dimension
        self.txt_proj = nn.Linear(txt_dim, dim) if txt_dim != dim else nn.Identity()

    def forward(self, sel_tokens, q_vec=None, targets=None, start_token_id=1, end_token_id=2):
        """
        Standard forward pass for training.
        """
        B, K, D = sel_tokens.shape
        mem = self.mem_proj(sel_tokens)
        
        # concatenate Question Vector to Visual Memory
        if q_vec is not None:
            q_mem = self.txt_proj(q_vec).unsqueeze(1)
            mem = torch.cat([q_mem, mem], dim=1)

        if targets is not None:
            # --- TRAINING PATH ---
            inp = targets[:, :-1]
            tgt = targets[:, 1:]
            
            # create Causal Mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(inp.size(1)).to(mem.device)
            
            # embeddings + Positional Encoding
            q = self.tok(inp) + self.pos[:, : inp.size(1)]
            
            # transformer Pass
            h = self.dec(q, mem, tgt_mask=tgt_mask)
            logits = self.out(h)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=0, # assuming 0 is <pad>
            )
            return logits, loss
        else:
            # fallback to greedy if no targets provided (mostly for debugging)
            return self.beam_search(sel_tokens, q_vec, beam_size=1, start_token_id=start_token_id, end_token_id=end_token_id)

    def beam_search(self, sel_tokens, q_vec, beam_size=3, start_token_id=1, end_token_id=2):
        """
        Inference using Beam Search.
        """
        B, K, D = sel_tokens.shape
        device = sel_tokens.device
        
        # prepare memory
        mem = self.mem_proj(sel_tokens)
        if q_vec is not None:
            q_mem = self.txt_proj(q_vec).unsqueeze(1)
            mem = torch.cat([q_mem, mem], dim=1) # (B, K+1, D)

        all_generated = []

        # iterate over batch items (Sample-by-sample generation)
        for b in range(B):
            # expand memory for beams: (Beam, K+1, D)
            curr_mem = mem[b].unsqueeze(0).repeat(beam_size, 1, 1) 
            
            # beams: List of (log_prob, tensor_seq)
            beams = [(0.0, torch.tensor([start_token_id], device=device, dtype=torch.long))]
            
            for _ in range(self.max_len - 1):
                candidates = []
                
                # Expand each beam
                for score, seq in beams:
                    if seq[-1] == end_token_id:
                        candidates.append((score, seq))
                        continue
                    
                    curr_input = seq.unsqueeze(0) # (1, L)
                    L = curr_input.size(1)
                    
                    # Embed & Positional
                    q = self.tok(curr_input) + self.pos[:, :L]
                    
                    # Slice memory for single beam forward pass
                    h_mem = curr_mem[0].unsqueeze(0) 
                    
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
                    h = self.dec(q, h_mem, tgt_mask=tgt_mask)
                    
                    # Get log probabilities of the last token
                    log_probs = F.log_softmax(self.out(h[:, -1]), dim=-1).squeeze(0)
                    
                    # Select top candidates
                    topk_probs, topk_idx = log_probs.topk(beam_size)
                    
                    for k in range(beam_size):
                        new_score = score + topk_probs[k].item()
                        new_seq = torch.cat([seq, topk_idx[k].unsqueeze(0)])
                        candidates.append((new_score, new_seq))
                
                # Prune beams
                ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = ordered[:beam_size]
                
                # Stop if all beams ended
                if all(b[1][-1] == end_token_id for b in beams):
                    break
            
            # Best sequence
            best_seq = beams[0][1]
            all_generated.append(best_seq)

        # Pad results
        max_len_gen = max(len(s) for s in all_generated)
        result = torch.zeros(B, max_len_gen, dtype=torch.long, device=device)
        for i, seq in enumerate(all_generated):
            result[i, :len(seq)] = seq
            
        return None, result