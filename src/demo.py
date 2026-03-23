import torch, yaml, json, cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import gradio as gr
import numpy as np
# import matplotlib.cm as cm
from matplotlib import pyplot as plt

# Import your project components
from lib.model.vqa import DiffVQAModel
from src.train import tokenize_questions

# ---------------------------------------------------------
# 1. Global Setup & Model Loading
# ---------------------------------------------------------
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading ALIGN-VQA on {DEVICE}...")

CONFIG_PATH = "configs/topk_16.yaml"
MODEL_PATH = "models/finetuned_last.pth"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

vocab_path = Path("models/vocab.json")
with open(vocab_path, "r") as f:
    loaded_vocab = json.load(f)
vocab = (loaded_vocab["stoi"], loaded_vocab["itos"])
num_classes = len(vocab[1])

model = DiffVQAModel(
    backbone=cfg.get("backbone"),
    text_encoder=cfg.get("text_encoder"),
    text_model_name=cfg.get("text_model_name"),
    text_dim=cfg.get("text_dim"),
    text_proj_dim=cfg.get("text_proj_dim"),
    text_finetune=False, 
    topk=cfg.get("topk"),
    num_classes=num_classes,
    max_ans_len=cfg.get("max_ans_len"),
    freeze_backbone=False 
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=False)
model.eval()

MIMIC_MEAN = [0.485, 0.456, 0.406]
MIMIC_STD = [0.229, 0.224, 0.225]

image_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=MIMIC_MEAN, std=MIMIC_STD)
])

# ---------------------------------------------------------
# 2. Heatmap Generation Helper
# ---------------------------------------------------------
def generate_heatmap_overlay(original_img, attention_weights):
    """
    Overlays attention weights onto the original image.
    Expects attention_weights to be a 1D tensor/array of spatial tokens (e.g., 196 for 14x14).
    """
    # 1. Reshape the 1D attention array into a 2D spatial grid
    # Assuming standard Swin/ViT output where N = H x W (e.g., 196 -> 14x14)
    grid_size = int(np.sqrt(attention_weights.shape[0]))
    attn_grid = attention_weights.reshape((grid_size, grid_size))
    
    # 2. Normalize attention to 0-1 range
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    
    # 3. Resize attention grid to match the original image size (224x224)
    original_img = original_img.resize((224, 224))
    attn_grid_resized = np.array(Image.fromarray(attn_grid).resize((224, 224), resample=Image.BICUBIC))
    
    # 4. Apply colormap (Jet/Inferno are standard for medical heatmaps)
    # cmap = cm.get_cmap('jet')
    cmap = plt.get_cmap('jet')
    heatmap = cmap(attn_grid_resized)[:, :, :3] * 255 # Drop alpha channel, scale to 255
    heatmap = heatmap.astype(np.uint8)
    
    # 5. Blend the original image and the heatmap
    original_array = np.array(original_img.convert("RGB"))
    alpha = 0.5 # Transparency of the heatmap
    overlay = cv2.addWeighted(original_array, 1 - alpha, heatmap, alpha, 0) if 'cv2' in globals() else \
              (original_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)
              
    return Image.fromarray(overlay)

# ---------------------------------------------------------
# 3. Inference Function
# ---------------------------------------------------------
def predict_and_visualize(ref_img, cur_img, question):
    if ref_img is None or cur_img is None or not question.strip():
        return "⚠️ Please upload both images and enter a question.", None

    try:
        img_ref_tensor = image_transforms(ref_img).unsqueeze(0).to(DEVICE)
        img_cur_tensor = image_transforms(cur_img).unsqueeze(0).to(DEVICE)
        tokens = tokenize_questions(model.text, [question], device=DEVICE)

        with torch.no_grad():
            out = model(img_ref_tensor, img_cur_tensor, tokens)

            if hasattr(model.head, 'beam_search'):
                _, preds_ids = model.head.beam_search(
                    out["sel_tokens"], q_vec=out["q_vec"], beam_size=3
                )
            else:
                _, preds_ids = model.head(out["sel_tokens"], q_vec=out["q_vec"])
            
            preds_ids = preds_ids.cpu().tolist()[0] 
            
            pred_tokens = []
            for tid in preds_ids:
                if tid == 2: break
                if tid > 2: pred_tokens.append(vocab[1][tid])
            answer_str = " ".join(pred_tokens)

            # =================================================================
            # 🚨 CRITICAL STEP: GRAB YOUR QDT ATTENTION WEIGHTS HERE 🚨
            # Look inside your `out` dictionary. You need the raw attention 
            # scores before the Top-K masking, or the spatial weight matrix.
            # Example: attention_tensor = out["qdt_attention"][0].cpu().numpy()
            # =================================================================
            
            # --- MOCK ATTENTION FOR NOW (Delete this and use your actual tensor) ---
            # attention_tensor = np.random.rand(196) # Simulating a 14x14 grid
            real_attention = out['heatmap'][0]
            attention_tensor = real_attention.detach().cpu().numpy().flatten()
            # -----------------------------------------------------------------------

            # Generate the visualization on the *Current* image
            heatmap_img = generate_heatmap_overlay(cur_img, attention_tensor)

            return answer_str, heatmap_img

    except Exception as e:
        return f"Error during inference: {str(e)}", None

# ---------------------------------------------------------
# 4. Gradio UI Layout
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="ALIGN-VQA Clinical Demo") as demo:
    
    gr.Markdown("# ALIGN-VQA: Longitudinal Medical VQA")
    gr.Markdown("Upload a historical reference scan and a current scan to evaluate disease progression. The model will output its prediction along with an attention heatmap")
    
    with gr.Row():
        with gr.Column():
            ref_image_input = gr.Image(type="pil", label="Reference Image (Past)")
        with gr.Column():
            cur_image_input = gr.Image(type="pil", label="Current Image (Present)")
            
    question_input = gr.Textbox(
        lines=2, 
        placeholder="e.g., What has changed compared to the reference image?", 
        label="Clinical Question"
    )
    
    submit_btn = gr.Button("Generate Answer & Heatmap", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="ALIGN-VQA Prediction", lines=5)
        with gr.Column(scale=1):
            heatmap_output = gr.Image(type="pil", label="QDT Difference Attention Map")
    
    submit_btn.click(
        fn=predict_and_visualize,
        inputs=[ref_image_input, cur_image_input, question_input],
        outputs=[answer_output, heatmap_output]
    )

if __name__ == "__main__":
    print("Launching Gradio interface with Heatmaps...")
    demo.launch(share=False, server_port=7860)