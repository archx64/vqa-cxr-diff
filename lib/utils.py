import logging, colorlog, json, nltk, os
import torch
import numpy as np

from logging.handlers import RotatingFileHandler
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor # Optional if you have java
from nltk.translate.meteor_score import meteor_score

# Constants for Neptune (Optional)
NEPTUNE_PROJECT = "DRIFT/medical-diff-vqa"
NEPTUNE_API_TOKEN = "" # Add your token if using Neptune

def run_nlg_evaluation(gts, res):
    """
    Calculates standard NLG metrics.
    """
    # Clean up previous runs
    if os.path.exists('gts.json'): os.remove('gts.json')
    if os.path.exists('res.json'): os.remove('res.json')

    with open('gts.json', 'w') as f: json.dump(gts, f)
    with open('res.json', 'w') as f: json.dump(res, f)

    coco = COCO('gts.json')
    coco_result = coco.loadRes('res.json')
    coco_eval = CustomCOCOEvalCap(coco, coco_result)
    
    # Suppress print output during eval
    coco_eval.evaluate()
    
    final_scores = {}
    for metric, score in coco_eval.eval.items():
        final_scores[metric] = score

    return final_scores

def collate(batch):
    """
    Custom collate function to handle image stacking and 
    NEW: label stacking for aux classification.
    """
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
            
    # Stack Images
    out["img_cur"] = torch.stack(out["img_cur"])
    out["img_ref"] = torch.stack(out["img_ref"])

    # Stack Token IDs (Targets)
    if "answer_ids" in out:
        out["answer_ids"] = torch.stack(out["answer_ids"])

    # --- CHANGED: Stack Disease Labels for Aux Loss ---
    if "cur_labels" in out:
        out["cur_labels"] = torch.stack(out["cur_labels"])
    if "ref_labels" in out:
        out["ref_labels"] = torch.stack(out["ref_labels"])
    # --------------------------------------------------

    # 'meta' and 'question' remain lists of strings/tuples
    return out

def make_loader(ds, bs, shuffle, num_workers=4):
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Improved speed
        collate_fn=collate,
    )

def setup_logging(log_file: str, console_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Silence noisy libs
    for lib in ["urllib3", "huggingface_hub", "timm", "neptune", "bravado", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.propagate = True

    # File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console Handler
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_white, bg_red",
        },
    )
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    return logger

class MeteorNLTK:
    def __init__(self):
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def compute_score(self, gts, res):
        scores = []
        for img_id in gts:
            references = [r.split() for r in gts[img_id]]
            hypothesis = res[img_id][0].split()
            score = meteor_score(references, hypothesis)
            scores.append(score)
        return np.mean(scores), scores

    def method(self):
        return "METEOR"

class CustomCOCOEvalCap(COCOEvalCap):
    def evaluate(self):
        imgIds = self.params["image_id"]
        gts = self.coco.imgToAnns
        res = self.cocoRes.imgToAnns

        # Format conversion
        gts = {i: [a["caption"] for a in gts[i]] for i in gts}
        res = {i: [a["caption"] for a in res[i]] for i in res}

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (MeteorNLTK(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        for scorer, method in scorers:
            # print(f"Computing {scorer.method()}...")
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                
        self.setEvalImgs()