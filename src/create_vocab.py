import argparse
import json
import yaml
from pathlib import Path
from lib.dataset import DiffVQADataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to your server_train.yaml")
    args = parser.parse_args()

    # Load config to get data paths
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Building vocabulary from: {cfg['train_pairs_csv']}")

    # Initialize dataset (this automatically builds the vocab)
    ds = DiffVQADataset(
        data_root=cfg["data_root"],
        pairs_csv=cfg["train_pairs_csv"],
        meta_csv=cfg["train_meta_csv"],
        split="train",
        vocab=None, # This forces a rebuild
        max_ans_len=100
    )

    # Prepare output directory
    output_path = Path("models/vocab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"Saving vocab to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({
            "stoi": ds.stoi,
            "itos": ds.itos
        }, f, indent=2)
    
    print(f"Success! Vocab size: {len(ds.itos)}")

if __name__ == "__main__":
    main()