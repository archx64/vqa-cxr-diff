import csv, logging
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# MIMIC Normalization
MIMIC_MEAN = [0.485, 0.456, 0.406]
MIMIC_STD = [0.229, 0.224, 0.225]

# 14 CheXpert Labels
DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def gray_to_rgb(img):
    if img.mode != "L":
        img = img.convert("L")
    return img.convert("RGB")


img_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Swin-Tiny standard input
        transforms.Lambda(gray_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(MIMIC_MEAN, MIMIC_STD),
    ]
)

logger = logging.getLogger(__name__)


class DiffVQADataset(Dataset):
    def __init__(
        self, data_root, pairs_csv, meta_csv, split="train", vocab=None, max_ans_len=100
    ):
        self.data_root = Path(data_root)
        # self.data_root = data_root
        self.max_ans_len = max_ans_len
        self.split = split

        # We perform extraction every time to ensure labels are linked correctly
        # You can cache this dictionary if startup is slow
        self.rows, self.study_to_path, self.study_to_labels = self._load_data(
            pairs_csv, meta_csv
        )

        if vocab is None:
            logger.info("Building Vocabulary...")
            all_answers_text = [self._norm(r["answer"]) for r in self.rows]
            word_counts = Counter(
                word for text in all_answers_text for word in text.split()
            )
            words = [word for word, count in word_counts.items() if count >= 1]
            self.itos = ["<pad>", "<start>", "<end>", "<unk>"] + sorted(words)
            self.stoi = {word: i for i, word in enumerate(self.itos)}
        else:
            self.stoi, self.itos = vocab

    def _load_data(self, pairs_csv, meta_csv):
        path_map = {}
        label_map = {}

        logger.info(f"Loading metadata from {meta_csv}...")
        with open(meta_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["study_id"]
                subj = row["subject_id"]

                # Extract Labels (14-dim vector)
                # Logic: 1.0 -> 1, everything else -> 0 (U-Zeros)
                lbls = []
                for d in DISEASE_LABELS:
                    val = row.get(d, "")
                    try:
                        is_pos = 1.0 if float(val) == 1.0 else 0.0
                    except:
                        is_pos = 0.0
                    lbls.append(is_pos)
                label_map[sid] = torch.tensor(lbls, dtype=torch.float32)

                # Path
                pfx = f"p{str(subj)[:2]}"
                pdir = self.data_root / pfx / f"p{subj}" / f"s{sid}"
                # Optimization: In a real run, check file existence lazily or cache it
                # For now, we assume standard MIMIC structure
                path_map[sid] = pdir

        rows = []
        with open(pairs_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["study_id"] in path_map and r["ref_id"] in path_map:
                    rows.append(r)

        return rows, path_map, label_map

    def _norm(self, s):
        return s.strip().lower().replace(".", "").replace(",", "")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cur_id = r["study_id"]
        ref_id = r["ref_id"]

        # Image Loading
        # Note: We need to find the exact JPG.
        # Since _load_data saved the directory, we find the file here.
        cur_dir = self.study_to_path[cur_id]
        ref_dir = self.study_to_path[ref_id]

        # Simple glob (assuming 1 image per study for simplicity, or picking first)
        try:
            cur_path = next(cur_dir.glob("*.jpg"))
            ref_path = next(ref_dir.glob("*.jpg"))
        except StopIteration:
            # Fallback if file missing (should not happen if data is clean)
            cur_path = next(self.data_root.glob("**/*.jpg"))
            ref_path = cur_path

        img_cur = img_tf(Image.open(cur_path))
        img_ref = img_tf(Image.open(ref_path))

        # Labels
        cur_labels = self.study_to_labels.get(cur_id, torch.zeros(14))
        ref_labels = self.study_to_labels.get(ref_id, torch.zeros(14))

        # Text
        q = r["question"].strip().lower()
        a_text = self._norm(r["answer"])
        tokens = ["<start>"] + a_text.split() + ["<end>"]
        answer_ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]
        padded = answer_ids[: self.max_ans_len] + [self.stoi["<pad>"]] * (
            self.max_ans_len - len(answer_ids)
        )

        return {
            "img_cur": img_cur,
            "img_ref": img_ref,
            "cur_labels": cur_labels,
            "ref_labels": ref_labels,
            "question_type": r.get("question_type", "unknown"),
            "question": q,
            "answer_ids": torch.tensor(padded, dtype=torch.long),
            "meta": (r["subject_id"], cur_id, ref_id),
        }
