import torch
from pathlib import Path
import random, os, json, yaml, argparse, logging
from tqdm import tqdm
from collections import defaultdict

from bert_score import score as bert_score

from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging, run_nlg_evaluation
from lib.model.vqa import DiffVQAModel
from src.train import tokenize_questions

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DRIFT-VQA V3 model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument("--bs", type=int, default=16, help="Test batch size")
    return parser.parse_args()


def get_bert_scores(gts_dict, res_list):
    """Helper function to calculate BERTScore for a set of predictions."""
    # Align predictions with ground truths using the image_id
    gt_map = {ann["image_id"]: ann["caption"] for ann in gts_dict["annotations"]}
    cands = []
    refs = []

    for r in res_list:
        cands.append(r["caption"])
        refs.append(gt_map[r["image_id"]])

    # Calculate BERTScore
    P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)

    return {
        "BERT-P": P.mean().item(),
        "BERT-R": R.mean().item(),
        "BERT-F1": F1.mean().item(),
    }


def seed_all(s=42):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    torch.backends.cudnn.benchmark = True


def main(args):
    setup_logging(log_file="logs/test.log", console_level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load Vocab
    vocab_path = Path("models/vocab.json")
    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    vocab = (loaded_vocab["stoi"], loaded_vocab["itos"])
    num_classes = len(vocab[1])

    # 3. Initialize Model (V3 Params)
    model = DiffVQAModel(
        backbone=cfg.get("backbone", "swin_tiny_patch4_window7_224"),
        text_encoder=cfg.get("text_encoder", "clinicalbert"),
        text_model_name=cfg.get("text_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_dim=cfg.get("text_dim", 768),
        text_proj_dim=cfg.get("text_proj_dim", 768),
        text_finetune=False,
        num_classes=num_classes,
        topk=cfg.get("topk", 64),
        max_ans_len=cfg.get("max_ans_len", 100),
        freeze_backbone=False,
    ).to(device)

    # 4. Load State Dict
    logger.info(f"Loading weights from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean load (ignore strict match for safety)
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Load Result: {msg}")
    model.eval()

    # 5. Prepare Data (V3 Dataset Signature)
    test_ds = DiffVQADataset(
        data_root=cfg["data_root"],
        pairs_csv=cfg["test_pairs_csv"],
        meta_csv=cfg["test_meta_csv"],
        split="test",
        vocab=vocab,
        max_ans_len=cfg.get("max_ans_len", 192),
    )

    test_loader = make_loader(test_ds, bs=args.bs, shuffle=False, num_workers=4)

    # 6. Inference Loop Setup
    # Dictionaries for overall evaluation
    overall_gts = {"info": {}, "images": [], "annotations": []}
    overall_res = []

    # Dictionaries to group results by question type
    gts_by_type = defaultdict(lambda: {"info": {}, "images": [], "annotations": []})
    res_by_type = defaultdict(list)

    sample_id = 0

    logger.info(f"Evaluating {len(test_ds)} samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            gt_ids = batch["answer_ids"].cpu()

            # Fetch question types (fallback to 'unknown' if not present in batch)
            q_types = batch.get("question_type", ["unknown"] * len(qs))

            tokens = tokenize_questions(model.text, qs, device=device)
            out = model(img_ref, img_cur, tokens)

            # --- USE BEAM SEARCH FOR HIGHER SCORES ---
            if hasattr(model.head, "beam_search"):
                _, preds_ids = model.head.beam_search(
                    out["sel_tokens"], q_vec=out["q_vec"], beam_size=3
                )
            else:
                _, preds_ids = model.head(out["sel_tokens"], q_vec=out["q_vec"])

            preds_ids = preds_ids.cpu().tolist()

            for i in range(len(qs)):
                q_type = q_types[i]

                # Decode Ground Truth
                gt_tokens = [vocab[1][tid] for tid in gt_ids[i].tolist() if tid > 2]
                gt_str = " ".join(gt_tokens)

                # Decode Prediction
                pred_tokens = []
                for tid in preds_ids[i]:
                    if tid == 2:
                        break  # <end>
                    if tid > 2:
                        pred_tokens.append(vocab[1][tid])
                pred_str = " ".join(pred_tokens)

                # Append to Overall Tracking
                overall_gts["images"].append({"id": sample_id})
                overall_gts["annotations"].append(
                    {
                        "image_id": sample_id,
                        "id": sample_id,
                        "question": qs[i],
                        "caption": gt_str,
                    }
                )
                overall_res.append({"image_id": sample_id, "caption": pred_str})

                # Append to Type-Specific Tracking
                gts_by_type[q_type]["images"].append({"id": sample_id})
                gts_by_type[q_type]["annotations"].append(
                    {
                        "image_id": sample_id,
                        "id": sample_id,
                        "question": qs[i],
                        "caption": gt_str,
                    }
                )
                res_by_type[q_type].append({"image_id": sample_id, "caption": pred_str})

                sample_id += 1

    # 7. Metrics Evaluation & Output
    all_results_dict = {}

    if overall_res:
        # Evaluate Overall Set
        print("\n" + "=" * 40)
        print("FINAL TEST SCORES (OVERALL)")
        print("=" * 40)
        overall_scores = run_nlg_evaluation(overall_gts, overall_res)
        all_results_dict["overall"] = overall_scores
        for metric, val in overall_scores.items():
            print(f"{metric:10}: {val:.4f}")

        # Evaluate Per Question Type
        for q_type in sorted(gts_by_type.keys()):
            if len(res_by_type[q_type]) > 0:
                print("\n" + "-" * 40)
                print(
                    f"SCORES FOR TYPE: {q_type.upper()} (N={len(res_by_type[q_type])})"
                )
                print("-" * 40)
                type_scores = run_nlg_evaluation(
                    gts_by_type[q_type], res_by_type[q_type]
                )
                all_results_dict[q_type] = type_scores
                for metric, val in type_scores.items():
                    print(f"{metric:10}: {val:.4f}")

        # Save comprehensive results to JSON
        with open("test_results_swin.json", "w") as f:
            json.dump(
                {"scores": all_results_dict, "predictions_sample": overall_res[:50]},
                f,
                indent=2,
            )
    else:
        logger.error("No results generated.")


if __name__ == "__main__":
    seed_all(42)
    main(parse_args())
