import argparse
import random
from pathlib import Path
import logging
import yaml
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Neptune
try:
    import neptune
except ImportError:
    print("Neptune not installed. Logging disabled.")

from tqdm import tqdm

from lib.dataset import DiffVQADataset
from lib.utils import (
    make_loader,
    setup_logging,
    run_nlg_evaluation,
    NEPTUNE_API_TOKEN,
    NEPTUNE_PROJECT,
)
from lib.model.vqa import DiffVQAModel
from lib.losses import heatmap_kl, info_nce_token_sets
from lib.negate import negate_question

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger()
run = None
model_name = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DRIFT-VQA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--train_pairs_csv", type=str)
    parser.add_argument("--train_meta_csv", type=str)
    parser.add_argument("--val_pairs_csv", type=str)
    parser.add_argument("--val_meta_csv", type=str)
    parser.add_argument("--test_pairs_csv", type=str)
    parser.add_argument("--test_meta_csv", type=str)

    parser.add_argument("--backbone", type=str)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")

    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--text_model_name", type=str)
    parser.add_argument("--text_finetune", action="store_true")
    parser.add_argument("--text_dim", type=int)
    parser.add_argument("--text_proj_dim", type=int)
    parser.add_argument("--max_ans_len", type=int)
    parser.add_argument("--topk", type=int, default=64)

    parser.add_argument("--bs", type=int)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--epochs_mrm", type=int)
    parser.add_argument("--epochs_warmup", type=int)
    parser.add_argument("--epochs_vqa", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float)

    parser.add_argument("--main_loss_weight", type=float)
    parser.add_argument("--lambda_mrm", type=float)
    parser.add_argument("--lambda_cf", type=float)
    parser.add_argument("--lambda_cls", type=float)
    parser.add_argument("--lambda_gate", type=float)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--ablation_no_direction", action="store_true")
    parser.add_argument("--use_cida", action="store_true")

    args = parser.parse_args()

    # --- ERROR CHECKING ---
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Config file not found at {config_path.absolute()}"
        )

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in cfg.items():
        setattr(args, k, v)

    # Validate essential paths are present
    if args.train_meta_csv is None:
        raise ValueError(
            "Error: 'train_meta_csv' is missing from config and command line args."
        )

    return args


def seed_all(s=42):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def tokenize_questions(text_model, batch_questions, device=None):
    enc = text_model.tokenize(batch_questions)
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def run_epoch(
    stage,
    model,
    loader,
    optimizer,
    scaler,
    device,
    lambda_gate=0.01,
    lambda_mrm=0.1,
    lambda_cls=0.5,
    lambda_cf=0.1,
    main_loss_weight=1.0,
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Stage {stage}", leave=False)

    # Loss for Multi-label classification (Novelty 2)
    bce_loss = nn.BCEWithLogitsLoss()

    for i, batch in enumerate(pbar):
        img_cur = batch["img_cur"].to(device)
        img_ref = batch["img_ref"].to(device)
        qs = batch["question"]
        y_seq = batch["answer_ids"].to(device)

        # New Labels from Dataset
        lbl_cur = batch["cur_labels"].to(device)
        lbl_ref = batch["ref_labels"].to(device)

        tokens = tokenize_questions(model.text, qs, device=device)

        # Counterfactual inputs logic
        if lambda_cf > 0:
            qs_cf = [negate_question(q) for q in qs]
            tokens_cf = tokenize_questions(model.text, qs_cf, device=device)
            # Swap images: If we negate question, Ref becomes Cur?
            # Or simplified CF: Just verify consistency.
            img_cur_cf, img_ref_cf = img_ref, img_cur
        else:
            tokens_cf, img_cur_cf, img_ref_cf = None, None, None

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            # --- Main Forward ---
            out = model(img_ref, img_cur, tokens)

            # 1. Generation Loss
            _, loss_gen = model.head(
                out["sel_tokens"], targets=y_seq, q_vec=out["q_vec"]
            )

            # 2. Auxiliary Classification Loss (Novelty 2)
            # Force backbone to predict disease presence
            loss_cls_cur = bce_loss(out["logits_cur"], lbl_cur)
            loss_cls_ref = bce_loss(out["logits_ref"], lbl_ref)
            loss_cls_total = (loss_cls_cur + loss_cls_ref) / 2

            # 3. Auxiliary Structure Losses
            loss_mrm = out["loss_mrm"]
            loss_gate = out["gate_l1"]

            # 4. Counterfactual Loss
            loss_cf_combined = 0.0
            loss_hkl = 0.0
            loss_nce = 0.0
            if lambda_cf > 0:
                out_cf = model(img_ref_cf, img_cur_cf, tokens_cf)
                loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
                loss_nce = info_nce_token_sets(out["sel_tokens"], out_cf["sel_tokens"])
                loss_cf_combined = loss_hkl + loss_nce

            # Combine
            if stage == "mrm":
                loss = loss_mrm
            elif stage == "warmup":
                loss = (
                    (main_loss_weight * loss_gen)
                    + (lambda_mrm * loss_mrm)
                    + (lambda_cls * loss_cls_total)
                )
            else:  # Full Training
                loss = (
                    (main_loss_weight * loss_gen)
                    + (lambda_mrm * loss_mrm)
                    + (lambda_cls * loss_cls_total)
                    + (lambda_cf * loss_cf_combined)
                    + (lambda_gate * loss_gate)
                )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if run and (i % 50 == 0):
            run[f"train/{stage}/loss"].append(loss.item())
            run[f"train/{stage}/loss_mrm"].append(loss_mrm.item())
            run[f"train/{stage}/loss_cls_ref"].append(loss_cls_ref.item())
            run[f"train/{stage}/loss_cls_cur"].append(loss_cls_cur.item())
            run[f"train/{stage}/loss_cls_total"].append(loss_cls_total.item())
            run[f"train/{stage}/loss_gate"].append(loss_gate.item())
            run[f"train/{stage}/loss_gen"].append(
                loss_gen.item() if torch.is_tensor(loss_gen) else loss_gen
            )
            if lambda_cf > 0:
                run[f"train/{stage}/loss_hkl"].append(loss_hkl)
                run[f"train/{stage}/loss_nce"].append(loss_nce)
                run[f"train/{stage}/loss_cf_combined"].append(loss_cf_combined)

        pbar.set_description(f"Batch Loss: {loss.item():.4f}")

    return running_loss / len(loader)


def evaluate(model, loader, device, vocab):
    model.eval()
    gts = {"info": {}, "images": [], "annotations": []}
    res = []
    sample_id = 0
    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in pbar:
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            gt_ids = batch["answer_ids"].cpu()

            tokens = tokenize_questions(model.text, qs, device=device)
            out = model(img_ref, img_cur, tokens)

            # Use Beam Search for better quality
            _, preds_ids = model.head.beam_search(
                out["sel_tokens"], q_vec=out["q_vec"], beam_size=3
            )
            preds_ids = preds_ids.cpu().tolist()

            for i in range(len(qs)):
                gt_toks = [vocab[1][tid] for tid in gt_ids[i].tolist() if tid > 2]
                pred_toks = []
                for tid in preds_ids[i]:
                    if tid == 2:
                        break
                    if tid > 2:
                        pred_toks.append(vocab[1][tid])

                gts["images"].append({"id": sample_id})
                gts["annotations"].append(
                    {
                        "image_id": sample_id,
                        "id": sample_id,
                        "caption": " ".join(gt_toks),
                    }
                )
                res.append({"image_id": sample_id, "caption": " ".join(pred_toks)})
                sample_id += 1

    val_scores = {}
    if res:
        print("\n--- Validation Metrics ---")
        val_scores = run_nlg_evaluation(gts, res)
        if run:
            for k, v in val_scores.items():
                run[f"val/metrics/{k}"].append(v)
    return val_scores


def main(args):
    global logger, run, model_name
    if not logger.hasHandlers():
        logger = setup_logging("logs/train.log")

    model_name = f"SWIN_{args.topk}_cls-{args.lambda_cls}_gate-{args.lambda_gate}_cf-{args.lambda_cf}_usecida-{args.use_cida}"
    try:
        run = neptune.init_run(
            project=NEPTUNE_PROJECT, name=model_name, api_token=NEPTUNE_API_TOKEN
        )
        run["parameters"] = vars(args)
    except:
        run = None

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = DiffVQADataset(
        args.data_root,
        args.train_pairs_csv,
        args.train_meta_csv,
        split="train",
        max_ans_len=args.max_ans_len,
    )
    vocab = (train_ds.stoi, train_ds.itos)
    val_ds = DiffVQADataset(
        args.data_root,
        args.val_pairs_csv,
        args.val_meta_csv,
        split="val",
        vocab=vocab,
        max_ans_len=args.max_ans_len,
    )

    train_loader = make_loader(
        train_ds, args.bs, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        val_ds, args.bs, shuffle=False, num_workers=args.num_workers
    )

    # Model
    model = DiffVQAModel(
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        pretrained_weights_path=args.ckpt,
        text_dim=args.text_dim,
        text_proj_dim=args.text_proj_dim,
        text_finetune=args.text_finetune,
        topk=args.topk,
        num_classes=len(vocab[1]),
        max_ans_len=args.max_ans_len,
        mask_ratio=args.mask_ratio,
        use_cida=args.use_cida,
    ).to(device)

    # Differential LR
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name or "drs" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    # scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scaler = torch.amp.GradScaler(device=device, enabled=(device.type == "cuda"))

    # Training Loop
    for ep in range(int(args.epochs_mrm)):
        run_epoch(
            "mrm",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=args.lambda_mrm,
            lambda_cls=0.0,
            lambda_cf=0.0,
            lambda_gate=args.lambda_gate,
        )

    for ep in range(int(args.epochs_warmup)):
        run_epoch(
            "warmup",
            model,
            train_loader,
            opt,
            scaler,
            device,
            main_loss_weight=args.main_loss_weight,
            lambda_mrm=args.lambda_mrm,
            lambda_cls=args.lambda_cls,
            lambda_cf=0.0,
            lambda_gate=args.lambda_gate,
        )

    scheduler = CosineAnnealingLR(opt, T_max=int(args.epochs_vqa), eta_min=1e-8)
    best_cider = 0.0

    logger.info("Starting VQA Training...")
    for ep in range(int(args.epochs_vqa)):
        run_epoch(
            "vqa",
            model,
            train_loader,
            opt,
            scaler,
            device,
            main_loss_weight=args.main_loss_weight,
            lambda_mrm=0.0,
            lambda_cls=args.lambda_cls,
            lambda_cf=args.lambda_cf,
            lambda_gate=args.lambda_gate,
        )

        scheduler.step()

        scores = evaluate(model, val_loader, device, vocab)
        score_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
        logger.info(f"\nEpoch {ep+1} Validation Scores:\n{score_str}\n")
        cider = scores.get("CIDEr", 0.0)

        if cider > best_cider:
            best_cider = cider
            # Save weights AND config
            state = {
                "model": model.state_dict(),
                "config": vars(args),
                "best_cider": best_cider,
            }
            torch.save(state, f"models/{model_name}_best.pth")
            logger.info(f"*** new best model saved (CIDEr: {best_cider:.4f}) ***")

    torch.save(state, f"models/{model_name}_last.pth")

    if run:
        run.stop()


if __name__ == "__main__":
    args = parse_args()
    main(args)
