import os
# Enable OpenEXR codec for OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image

def convert_exr_to_indexed_png(file_path, output_filename):
    # --- STEP 1: LOAD EXR IMAGE ---
    print(f"Loading: {file_path}")
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print("Error: Could not load image. Check path.")
        return

    # Handle channels (Keep only RGB, drop Alpha if present)
    if len(img.shape) == 3 and img.shape[2] >= 3:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img

    print(f"Processing image shape: {img_rgb.shape}")

    # --- STEP 2: CONVERT COLORS TO INTEGER LABELS (0, 1, 2...) ---
    # Flatten to list of pixels (N, 3)
    pixels = img_rgb.reshape(-1, 3)
    
    # Find unique colors. 'inverse' is our new integer label map.
    unique_colors, inverse = np.unique(pixels, axis=0, return_inverse=True)
    num_classes = len(unique_colors)
    
    print(f"Found {num_classes} unique semantic classes.")
    
    # Reshape back to (H, W) to get our final label map
    label_map = inverse.reshape(img_rgb.shape[0], img_rgb.shape[1]).astype(np.uint8)

    # --- STEP 3: CREATE PALETTE & SAVE USING PILLOW ---
    # Create a PIL image in 'P' (Palette) mode
    pil_img = Image.fromarray(label_map, mode='P')

    # Generate a distinct color palette
    # We define a standard list of colors for the first few IDs
    # Format: [R, G, B,  R, G, B, ...]
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

    # If we have more classes than our base palette, generate random colors
    current_palette_len = len(base_palette) // 3
    if num_classes > current_palette_len:
        import random
        for _ in range(num_classes - current_palette_len):
            base_palette.extend([random.randint(50, 255) for _ in range(3)])

    # Pad the palette to exactly 256 colors (Required by PNG spec)
    # (256 * 3 = 768 integers total)
    full_palette = base_palette + [0, 0, 0] * (256 - len(base_palette) // 3)
    
    # Apply the palette
    pil_img.putpalette(full_palette)

    # Save
    pil_img.save(output_filename)
    print(f"\nSuccess! Saved to: {output_filename}")
    print(" -> This file looks colorful but contains raw integers (0, 1, 2...) when loaded as data.")
