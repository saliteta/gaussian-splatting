import argparse
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image, write_png, ImageReadMode
from pathlib import Path

CLASS_STRENGTHS = {
    0: 1.0,
    1: 5.0,
    2: 0.0,
    3: 1.0,
}

def create_soft_semantic_tensor(img_tensor: torch.Tensor, kernel_size: int = 31, sigma: float = 5.0) -> torch.Tensor:
    """
    Takes a grayscale label tensor [H, W] or [1, H, W], extracts classes 0, 1, 3, 
    applies a Gaussian blur, and normalizes the probabilities.
    
    Returns:
        normalized_maps: A float tensor of shape [3, H, W] representing the soft probabilities.
    """
    # Ensure shape is [H, W] by dropping the channel dimension if it exists
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)

    # Cast to float for mathematical operations
    img_tensor = img_tensor.to(torch.float32) 


    # 1. Decompose into 3 separate maps (ignoring class 2)
    # Shape of each map: [H, W]
    map_0 = (img_tensor == 0.0).float() * CLASS_STRENGTHS[0]
    map_1 = (img_tensor == 1.0).float() * CLASS_STRENGTHS[1]
    map_2 = (img_tensor == 3.0).float() * CLASS_STRENGTHS[3]


    # Shape of stacked_maps: [3, H, W]
    stacked_maps = torch.stack([map_0, map_1, map_2], dim=0)

    # 2. Apply Gaussian Filter
    blurred_maps = F.gaussian_blur(
        stacked_maps, 
        kernel_size=[kernel_size, kernel_size], 
        sigma=[sigma, sigma]
    )

    # 3. Normalize so the 3 channels sum to 1.0 at each pixel
    channel_sum = blurred_maps.sum(dim=0, keepdim=True)
    
    # Add epsilon to prevent division by zero in regions that were entirely class '2'
    epsilon = 1e-8
    normalized_maps = blurred_maps / (channel_sum + epsilon)

    return normalized_maps


def process_semantic_folder(input_dir: str, output_dir: str, kernel_size: int = 31, sigma: float = 5.0):
    """
    Iterates through all PNGs in a folder, processes them into soft semantic RGB maps, 
    and saves them to an output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Grab all PNG files in the input folder
    image_files = list(input_path.glob('*.png'))
    
    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    for img_file in image_files:
        # Read the image directly into a PyTorch tensor (Shape: [1, H, W], dtype: uint8)
        # Using ImageReadMode.GRAY ensures we only get one channel
        raw_tensor = read_image(str(img_file), mode=ImageReadMode.UNCHANGED)
        
        # Pass the tensor to our core function
        soft_tensor = create_soft_semantic_tensor(raw_tensor, kernel_size, sigma)
        
        # Scale from [0.0, 1.0] to [0, 255] and cast to uint8 for saving
        output_tensor = (soft_tensor * 255).to(torch.uint8)
        
        # Construct the output filename and save
        out_file = output_path / f"{img_file.stem}_soft.png"
        write_png(output_tensor, str(out_file))
        
        print(f"Processed and saved: {out_file.name}")

    print("Sequential processing complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate soft semantic feature maps from label PNGs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input label PNGs")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save soft semantic maps (*_soft.png)")
    parser.add_argument("--kernel_size", type=int, default=7, help="Gaussian blur kernel size (default: 31)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur sigma (default: 5.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_semantic_folder(
        args.input_dir,
        args.output_dir,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
    )
