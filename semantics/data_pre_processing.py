"""
We convert the .exr files to PNG file:
Contain four value: 1,2,3,4
Color map:

    base_palette = [
        0, 0, 0,        # ID 0: Black (Background)
        255, 0, 0,      # ID 1: Red
        0, 255, 0,      # ID 2: Green
        0, 0, 255,      # ID 3: Blue
        255, 255, 0,    # ID 4: Yellow
        0, 255, 255,    # ID 5: Cyan
        255, 0, 255,    # ID 6: Magenta
        255, 165, 0,    # ID 7: Orange
        128, 0, 128,    # ID 8: Purple
    ]

    0 seemms to be ground
    1 seems to be trunk
    2 seems to be nothing
    3 seems to be tree
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Enable OpenEXR (Must be before importing cv2 if specific builds require it, 
# but inside the function for multiprocessing safety usually helps)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def process_single_file(file_info):
    """
    Worker function to process a single EXR file.
    Args:
        file_info (tuple): (input_path, output_path)
    """
    input_path, output_path = file_info

    # 1. Load Image
    # IMREAD_UNCHANGED is crucial for EXR
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return f"Error reading {input_path}"

    # 2. Extract ID Channel (Optimization)
    # Instead of unique() on (H,W,3) which is slow, we use just Channel 0.
    # We saw earlier that Channel 0 has distinct values [0, 0.5, 0.89, 1.0].
    # This is sufficient to distinguish classes.
    if len(img.shape) == 3:
        # Take the first channel (Blue/Channel 0)
        data_channel = img[:, :, 0]
    else:
        data_channel = img

    # 3. Fast Mapping (Quantization)
    # Convert floats to a robust signature to avoid slow sorting
    # We multiply by 255 and round to get stable integers (0, 128, 228, 255 etc)
    # This is faster than np.unique on floats
    data_int = (data_channel * 255).round().astype(np.uint8)

    # 4. Map to 0, 1, 2, 3
    # unique() on a 1D flat array of uint8 is extremely fast
    unique_vals, inverse = np.unique(data_int, return_inverse=True)
    
    # Reshape back to image dimensions
    label_map = inverse.reshape(data_int.shape).astype(np.uint8)

    # 5. Save as Indexed PNG
    pil_img = Image.fromarray(label_map, mode='P')

    # Define Palette (0=Black, 1=Red, 2=Green, 3=Blue, etc.)
    base_palette = [
        0, 0, 0,        # ID 0: Ground (Black)
        255, 0, 0,      # ID 1: Trunk (Red)
        0, 255, 0,      # ID 2: Nothing (Green) - *Adjust based on your preference*
        0, 0, 255,      # ID 3: Tree (Blue)
        255, 255, 0,    # ID 4: Yellow
        0, 255, 255,    # ID 5: Cyan
    ]
    
    # Pad palette to 256 colors (768 integers)
    full_palette = base_palette + [0, 0, 0] * (256 - len(base_palette) // 3)
    pil_img.putpalette(full_palette)

    pil_img.save(output_path)
    return None # Success

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_segmentations_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path relative to parent of raw_segmentations_path")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), 
                        help="Number of parallel processes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser()

    # Setup paths
    raw_path = args.raw_segmentations_path
    
    # Logic to place output folder relative to parent
    # Example: /data/synthetic/raw -> /data/synthetic/output_name
    parent_dir = os.path.dirname(raw_path.rstrip(os.sep))
    output_dir = os.path.join(parent_dir, args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Gather files
    files_to_process = []
    print(f"Scanning {raw_path}...")
    
    for f in os.listdir(raw_path):
        if f.endswith("species.exr"):
            in_file = os.path.join(raw_path, f)
            # Create corresponding output filename
            out_name = f.replace("species.exr", "species.png")
            out_file = os.path.join(output_dir, out_name)
            files_to_process.append((in_file, out_file))

    print(f"Found {len(files_to_process)} EXR files. Starting batch processing with {args.workers} workers...")

    # Run in Parallel
    # Using ProcessPoolExecutor allows bypassing the Python GIL for CPU-bound tasks like this
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        results = list(tqdm(
            executor.map(process_single_file, files_to_process), 
            total=len(files_to_process),
            desc="Converting"
        ))

    # Check for errors
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for err in errors:
            print(err)
    else:
        print("\nDone. All files converted successfully.")

