import os
import pandas as pd
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Change these to match your actual paths
CSV_PATH = "D:/MIMIC-CXR-JPG/mimic_pair_questions_test.csv"       # The pairs_csv you feed into DiffVQADataset
MIMIC_DATA_ROOT = "D:/MIMIC-CXR-JPG/" # The root folder containing the 'p10' folders
OUTPUT_DIR = "demo_images"
MIMIC_META_CSV = "D:/MIMIC-CXR-JPG/mimic_all_test.csv"
NUM_PAIRS_TO_EXTRACT = 5
# ---------------------

def get_exact_pa_image_path(data_root, subject_id, study_id, dicom_id):
    """
    Constructs the exact path to the specific PA dicom image.
    """
    root = Path(data_root)
    subj_str = str(subject_id)
    sid_str = str(study_id)
    
    pfx = f"p{subj_str[:2]}"
    # Instead of globbing the first image, we target the specific PA dicom_id
    img_path = root / pfx / f"p{subj_str}" / f"s{sid_str}" / f"{dicom_id}.jpg"
    
    if img_path.exists():
        return img_path
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load official MIMIC metadata to find PA views
    print(f"Loading MIMIC metadata from {MIMIC_META_CSV} to filter PA views...")
    mimic_meta = pd.read_csv(MIMIC_META_CSV)
    
    # Filter for PA views only
    pa_meta = mimic_meta[mimic_meta['view'] == 'postero-anterior']
    
    # Create a dictionary mapping: study_id -> dicom_id (the specific PA image filename)
    # .first() is used just in case a single study has two PA scans
    pa_dict = pa_meta.groupby('study_id')['dicom_id'].first().to_dict()
    
    # 2. Load your test pairs
    print(f"Loading test pairs from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    extracted_count = 0
    
    for idx, row in df.iterrows():
        if extracted_count >= NUM_PAIRS_TO_EXTRACT:
            break
            
        subject_id = row.get("subject_id")
        ref_id = row.get("ref_id")        
        cur_id = row.get("study_id")      
        question = row.get("question", "Unknown question")
        answer = row.get("answer", "Unknown answer") 
        q_type = row.get("question_type", "unknown")
        
        # Filter 1: Only look at Difference questions
        if str(q_type).lower() != "difference":
            continue
            
        if not all([subject_id, ref_id, cur_id]):
            continue
            
        # Filter 2: BOTH the reference and current study MUST have a PA view available
        if ref_id not in pa_dict or cur_id not in pa_dict:
            continue
            
        # Get the specific dicom_id for the PA images
        ref_dicom_id = pa_dict[ref_id]
        cur_dicom_id = pa_dict[cur_id]
        
        # Get exact paths
        ref_img_path = get_exact_pa_image_path(MIMIC_DATA_ROOT, subject_id, ref_id, ref_dicom_id)
        cur_img_path = get_exact_pa_image_path(MIMIC_DATA_ROOT, subject_id, cur_id, cur_dicom_id)
        
        if ref_img_path and cur_img_path:
            # Create a dedicated folder for this pair
            pair_dir = os.path.join(OUTPUT_DIR, f"pair_{extracted_count + 1}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # Copy images
            shutil.copy(ref_img_path, os.path.join(pair_dir, "reference_image.jpg"))
            shutil.copy(cur_img_path, os.path.join(pair_dir, "current_image.jpg"))
            
            # Write a text file with the question and ground truth answer
            with open(os.path.join(pair_dir, "clinical_context.txt"), "w") as f:
                f.write(f"Question Type: {q_type}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Ground Truth Answer: {answer}\n")
            
            print(f"Successfully extracted Pair {extracted_count + 1} (Subject: {subject_id}) - PA Views Confirmed")
            extracted_count += 1

    print(f"\nDone! Extracted {extracted_count} strict PA-to-PA pairs into the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()