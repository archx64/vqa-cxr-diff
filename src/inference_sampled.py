import torch
import argparse
import yaml
import json
import pandas as pd
import os
import random
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import logging
import textwrap
from tqdm import tqdm

# Import V3 project components
from lib.model.vqa import DiffVQAModel
from lib.dataset import MIMIC_MEAN, MIMIC_STD, gray_to_rgb
from src.train import tokenize_questions
from lib.utils import setup_logging

logger = logging.getLogger(__name__)

# --- Swin-Tiny requires strict 224x224 input ---
inference_img_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(gray_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(MIMIC_MEAN, MIMIC_STD),
    ]
)

TOTAL_SAMPLES = 100

# Define the target categories
CATEGORIES = [
    "difference",
    "presence",
    "abnormality",
    "location",
    "level",
    "view",
    "type"
]

def create_heatmap(heatmap_tensor, img_size):
    """
    Resizes the attention map to image size.
    """
    try:
        if heatmap_tensor.ndim == 3:
            heatmap_tensor = heatmap_tensor.mean(dim=0)
            
        heatmap = heatmap_tensor.cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        h_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        heatmap_resized = F.interpolate(
            h_tensor,
            size=(img_size[1], img_size[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        
        return heatmap_resized
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return np.zeros((img_size[1], img_size[0]))

def get_balanced_samples(pairs_csv_path, meta_csv_path, data_root, total_samples=100):
    """
    Returns a list of samples evenly distributed across the defined categories.
    """
    logger.info(f"Loading metadata and building balanced dataset...")
    df_pairs = pd.read_csv(pairs_csv_path)
    df_meta = pd.read_csv(meta_csv_path)

    # Calculate how many samples we need per category
    samples_per_category = int(np.ceil(total_samples / len(CATEGORIES)))
    logger.info(f"Targeting ~{samples_per_category} samples per category.")

    # Apply heuristic to generate question_type if it doesn't exist
    if 'question_type' not in df_pairs.columns:
        logger.warning("'question_type' column missing. Applying text heuristics...")
        conditions = [
            df_pairs['question'].str.lower().str.contains('difference|change|compare'),
            df_pairs['question'].str.lower().str.contains('where|location|side'),
            df_pairs['question'].str.lower().str.contains('abnormality|abnormal|finding'),
            df_pairs['question'].str.lower().str.contains('level|severity'),
            df_pairs['question'].str.lower().str.contains('view|projection'),
            df_pairs['question'].str.lower().str.contains('type')
        ]
        choices = ['difference', 'location', 'abnormality', 'level', 'view', 'type']
        df_pairs['question_type'] = np.select(conditions, choices, default='presence')

    # Map Study ID to Path
    id_to_path = {}
    for _, row in df_meta.iterrows():
        subj = str(row["subject_id"])
        stud = str(row["study_id"])
        dicom = str(row["dicom_id"])
        p_group = "p" + subj[:2]
        rel_path = os.path.join(p_group, "p" + subj, "s" + stud, f"{dicom}.jpg")
        id_to_path[stud] = os.path.join(data_root, rel_path)

    balanced_samples = []

    # Sample evenly from each category
    for cat in CATEGORIES:
        df_cat = df_pairs[df_pairs['question_type'] == cat]
        
        if len(df_cat) == 0:
            logger.warning(f"No samples found for category: {cat}")
            continue
            
        # Shuffle the category dataframe
        df_cat = df_cat.sample(frac=1, random_state=42).reset_index(drop=True)
        
        cat_samples_collected = 0
        for idx, row in df_cat.iterrows():
            if cat_samples_collected >= samples_per_category:
                break
                
            cur_id = str(row["study_id"])
            ref_id = str(row["ref_id"])
            
            if cur_id in id_to_path and ref_id in id_to_path:
                cur_path = id_to_path[cur_id]
                ref_path = id_to_path[ref_id]
                if os.path.exists(cur_path) and os.path.exists(ref_path):
                    balanced_samples.append({
                        "index": idx,
                        "img_cur": cur_path,
                        "img_ref": ref_path,
                        "question": str(row["question"]),
                        "answer_gt": str(row["answer"]),
                        "study_id": cur_id,
                        "question_type": cat # Save type for folder routing
                    })
                    cat_samples_collected += 1

    random.shuffle(balanced_samples) # Shuffle the final combined list
    return balanced_samples[:105] # Cap at exactly 105 (15 * 7)

def visualize_and_save(sample, predicted_answer, output, img_ref_pil, img_cur_pil, base_save_dir):
    """
    Visualizes the Attention Heatmap and routes it to the correct category folder.
    """
    att_map = output["heatmap"].squeeze(0) 
    
    img_size = img_ref_pil.size
    heatmap_overlay = create_heatmap(att_map, img_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    axes[0].imshow(img_ref_pil, cmap="gray")
    axes[0].set_title("Reference Image\n(Previous State)")
    axes[0].axis("off")

    axes[1].imshow(img_cur_pil, cmap="gray")
    axes[1].imshow(heatmap_overlay, cmap="jet", alpha=0.4) 
    axes[1].set_title("Current Image + Attention\n(Red = Region of Interest)")
    axes[1].axis("off")

    w_q = "\n".join(textwrap.wrap(f"Q: {sample['question']}", width=60))
    w_gt = "\n".join(textwrap.wrap(f"GT: {sample['answer_gt']}", width=60))
    w_pred = "\n".join(textwrap.wrap(f"Pred: {predicted_answer}", width=60))
    
    text_str = f"{w_q}\n\n{w_gt}\n\n{w_pred}"

    plt.figtext(0.5, 0.02, text_str, ha="center", fontsize=11, 
                bbox={"facecolor": "white", "alpha": 0.9, "pad": 10})

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Route to category folder
    cat_dir = base_save_dir / sample["question_type"]
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = cat_dir / f"{sample['study_id']}.png"
    plt.savefig(save_path)
    plt.close(fig)

def main(args):
    global logger
    if not logger.hasHandlers():
        logger = setup_logging(log_file="logs/inference_sampled.log", console_level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Vocab
    vocab_path = Path("models/vocab.json")
    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    vocab = loaded_vocab["itos"]
    num_classes = len(vocab)

    # 3. Model
    model = DiffVQAModel(
        backbone=cfg.get("backbone", "swin_tiny_patch4_window7_224"),
        text_encoder=cfg.get("text_encoder", "clinicalbert"),
        text_model_name=cfg.get("text_model_name"),
        text_dim=cfg.get("text_dim"),
        text_proj_dim=cfg.get("text_proj_dim"),
        text_finetune=False, 
        topk=cfg.get("topk"),
        num_classes=num_classes,
        max_ans_len=cfg.get("max_ans_len"),
        mask_ratio=cfg.get("mask_ratio"),
        freeze_backbone=False 
    ).to(device)

    # 4. Load Weights
    logger.info(f"Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 5. Output Dir
    output_dir = Path("inference_figures/sampled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Get Balanced Samples
    samples = get_balanced_samples(
        cfg["test_pairs_csv"], 
        cfg["test_meta_csv"], 
        cfg["data_root"], 
        total_samples=TOTAL_SAMPLES
    )

    if not samples:
        logger.error("No samples found.")
        return

    logger.info(f"Processing {len(samples)} total balanced samples...")
    for sample in tqdm(samples):
        try:
            img_ref_pil = Image.open(sample["img_ref"]).convert("RGB")
            img_cur_pil = Image.open(sample["img_cur"]).convert("RGB")

            img_ref = inference_img_tf(img_ref_pil).unsqueeze(0).to(device)
            img_cur = inference_img_tf(img_cur_pil).unsqueeze(0).to(device)

            token_batch = tokenize_questions(model.text, [sample["question"]], device=device)

            with torch.no_grad():
                output = model(img_ref, img_cur, token_batch)
                
                if hasattr(model.head, 'beam_search'):
                    _, preds_ids = model.head.beam_search(
                        output["sel_tokens"], q_vec=output['q_vec'], beam_size=3
                    )
                else:
                    _, preds_ids = model.head(output["sel_tokens"], q_vec=output['q_vec'])

                preds_ids = preds_ids.cpu().tolist()[0]
                
                pred_tokens = []
                for token_id in preds_ids:
                    if token_id == 2: break
                    if token_id > 2: pred_tokens.append(vocab[token_id])
                
                predicted_answer = " ".join(pred_tokens)

            visualize_and_save(sample, predicted_answer, output, img_ref_pil, img_cur_pil, output_dir)

        except Exception as e:
            logger.error(f"Error on sample {sample['study_id']}: {e}")
            continue
            
    logger.info(f"Inference complete. Check the '{output_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)