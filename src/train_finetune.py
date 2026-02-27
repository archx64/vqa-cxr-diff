import torch
import torch.nn as nn
import yaml, argparse, logging, os
from pathlib import Path
from torch.optim import AdamW
from tqdm import tqdm

# Import your custom modules
from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging, run_nlg_evaluation
from lib.model.vqa import DiffVQAModel
from src.train import tokenize_questions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tune YAML config")
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to your fully trained model.pth")
    parser.add_argument("--save_dir", required=True, type=str, help="Directory to save checkpoints")
    return parser.parse_args()

def validate(model, val_loader, vocab, device):
    """Runs inference on the validation set and calculates NLG metrics."""
    model.eval()
    gts = {"info": {}, "images": [], "annotations": []}
    res = []
    sample_id = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='validating', leave=False):
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            gt_ids = batch["answer_ids"].cpu()

            tokens = tokenize_questions(model.text, qs, device=device)
            out = model(img_ref, img_cur, tokens)

            if hasattr(model.head, 'beam_search'):
                _, preds_ids = model.head.beam_search(
                    out["sel_tokens"], q_vec=out["q_vec"], beam_size=3
                )
            else:
                _, preds_ids = model.head(out["sel_tokens"], q_vec=out["q_vec"])
            
            preds_ids = preds_ids.cpu().tolist()

            for i in range(len(qs)):
                gt_tokens = [vocab[1][tid] for tid in gt_ids[i].tolist() if tid > 2]
                gt_str = " ".join(gt_tokens)

                pred_tokens = []
                for tid in preds_ids[i]:
                    if tid == 2: break 
                    if tid > 2: pred_tokens.append(vocab[1][tid])
                pred_str = " ".join(pred_tokens)

                gts["images"].append({"id": sample_id})
                gts["annotations"].append({
                    "image_id": sample_id, "id": sample_id, "question": qs[i], "caption": gt_str
                })
                res.append({"image_id": sample_id, "caption": pred_str})
                sample_id += 1

    # Calculate metrics
    scores = {}
    if res:
        scores = run_nlg_evaluation(gts, res)
        
    model.train() # Set back to training mode
    return scores

def main(args):
    logger = setup_logging(log_file="logs/finetune.log", console_level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Vocab
    vocab_path = Path("models/vocab.json")
    import json
    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    vocab = (loaded_vocab["stoi"], loaded_vocab["itos"])

    # 2. Load Data (Train AND Val splits)
    train_ds = DiffVQADataset(
        data_root=cfg["data_root"],
        pairs_csv=cfg["train_pairs_finetune_csv"], 
        meta_csv=cfg["train_meta_csv"], 
        split="train",
        vocab=vocab,
        max_ans_len=cfg.get("max_ans_len", 192)
    )
    train_loader = make_loader(train_ds, bs=cfg.get("bs", 16), shuffle=True, num_workers=4)

    val_ds = DiffVQADataset(
        data_root=cfg["data_root"],
        pairs_csv=cfg["val_pairs_csv"], # Assuming you have a validation CSV in config
        meta_csv=cfg["val_meta_csv"], 
        split="val",
        vocab=vocab,
        max_ans_len=cfg.get("max_ans_len", 192)
    )
    val_loader = make_loader(val_ds, bs=cfg.get("bs", 16), shuffle=False, num_workers=4)

    # 3. Initialize Model (with fixed max_ans_len)
    model = DiffVQAModel(
        backbone=cfg.get("backbone", "swin_tiny_patch4_window7_224"),
        text_encoder=cfg.get("text_encoder", "clinicalbert"),
        text_model_name=cfg.get("text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_dim=cfg.get("text_dim", 768),
        text_proj_dim=cfg.get("text_proj_dim", 768),
        num_classes=len(vocab[1]),
        topk=cfg.get("topk", 64),
        max_ans_len=cfg.get("max_ans_len", 192),
        freeze_backbone=True 
    ).to(device)

    # 4. Load the Pretrained Weights (with weights_only fix)
    logger.info(f"Loading generalist model weights from {args.pretrained_path}")
    checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)

    # 5. Fix the Freezing Attribute Error
    for param in model.drs.parameters():
        param.requires_grad = False
    logger.info("Visual backbone (drs) frozen. CIDA and Decoder will be fine-tuned.")

    # 6. Setup Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion_lang = nn.CrossEntropyLoss(ignore_index=vocab[0]["<pad>"])

    epochs = 10
    best_cider = 0.0
    os.makedirs(Path(args.save_dir), exist_ok=True)

    # 7. Training & Validation Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        logger.info(f"--- Starting Fine-Tuning Epoch {epoch+1}/{epochs} ---")
        
        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs} Training"):
            optimizer.zero_grad()
            
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            ans_ids = batch["answer_ids"].to(device)

            tokens = tokenize_questions(model.text, qs, device=device)
            out = model(img_ref, img_cur, tokens)
            
            # logits, _ = model.head(out['sel_tokens'], q_vec=out['q_vec'], targets=ans_ids)
            # loss = criterion_lang(logits.view(-1, logits.size(-1)), ans_ids.view(-1))
            logits, loss = model.head(out['sel_tokens'], q_vec=out['q_vec'], targets=ans_ids)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # --- Automated Validation Phase ---
        logger.info(f"Running Validation for Epoch {epoch+1}...")
        val_scores = validate(model, val_loader, vocab, device)
        
        if val_scores:
            current_cider = val_scores.get("CIDEr", 0.0)
            logger.info(f"Validation Scores - BLEU-4: {val_scores.get('Bleu_4', 0.0):.4f} | CIDEr: {current_cider:.4f}")
            
            # Save Best Model
            if current_cider > best_cider:
                best_cider = current_cider
                save_path = os.path.join(args.save_dir, "finetuned_best.pth")
                logger.info(f"New best CIDEr! Saving model to {save_path}")
                torch.save(model.state_dict(), save_path)
        else:
            logger.warning("Validation returned no scores.")
    save_path = os.path.join(args.save_dir, "finetuned_last.pth")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main(parse_args())