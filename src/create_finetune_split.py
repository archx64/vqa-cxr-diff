import pandas as pd
import numpy as np

def create_finetune_split(input_csv, output_csv, target_ratio=0.70):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Identify Difference vs. Non-Difference questions
    # If you don't have a 'question_type' column, use the heuristic:
    if "question_type" not in df.columns:
        diff_keywords = ['difference', 'change', 'compare', 'worsened', 'improved']
        df['question_type'] = np.where(
            df['question'].str.lower().str.contains('|'.join(diff_keywords)), 
            'difference', 
            'other'
        )
    
    # Separate the data
    df_diff = df[df['question_type'] == 'difference']
    df_other = df[df['question_type'] != 'difference']
    
    print(f"Original dataset: {len(df_diff)} Difference, {len(df_other)} Other.")
    
    # 2. Calculate new dataset sizes
    # We want df_diff to be 70% of the new dataset.
    # Total = len(df_diff) / 0.7
    # Other_Needed = Total * 0.3
    other_needed = int((len(df_diff) / target_ratio) * (1 - target_ratio))
    
    # 3. Sample the "Other" category to prevent catastrophic forgetting
    # We use sample() to randomly pick the exact number we need
    if other_needed < len(df_other):
        df_other_sampled = df_other.sample(n=other_needed, random_state=42)
    else:
        # If we don't have enough 'other', just use all of them (ratio will be >70%)
        df_other_sampled = df_other
        
    # 4. Combine and Shuffle
    df_finetune = pd.concat([df_diff, df_other_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"New Fine-tuning dataset: {len(df_diff)} Difference, {len(df_other_sampled)} Other.")
    print(f"Total size: {len(df_finetune)} pairs.")
    print(f"Actual Difference Ratio: {len(df_diff)/len(df_finetune):.1%}")
    
    # Save the new CSV
    df_finetune.to_csv(output_csv, index=False)
    print(f"Saved split to {output_csv}")

if __name__ == "__main__":
    create_finetune_split("E:/MIMIC-CXR-JPG-224x224/mimic_pair_questions_train.csv", "E:/MIMIC-CXR-JPG-224x224/mimic_pair_questions_finetune.csv")