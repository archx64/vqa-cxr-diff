import torch
import argparse

def count_params_in_pth(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint to CPU to save VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle different saving formats (extract the state_dict)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Assume it's a raw state_dict

    total_params = 0
    
    # Iterate through all saved layers and sum the number of elements (parameters)
    for layer_name, weight_tensor in state_dict.items():
        params_in_layer = weight_tensor.numel()
        total_params += params_in_layer
        
    print("-" * 40)
    print(f"Total Parameters in model: {total_params:,}")
    print("-" * 40)
    
    return total_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters in a .pth file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your .pth file")
    args = parser.parse_args()
    
    count_params_in_pth(args.model_path)