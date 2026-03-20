import numpy as np
from plyfile import PlyData, PlyElement

def downsample_ply_numpy(input_file, output_file, ratio=0.2):
    print(f"Loading PLY data from {input_file}...")
    # Read the PLY file
    plydata = PlyData.read(input_file)
    vertex_data = plydata['vertex'].data
    
    num_points = len(vertex_data)
    print(f"Original point count: {num_points}")
    
    # Calculate the number of points to keep (1/5th)
    num_to_keep = int(num_points * ratio)
    print(f"Downsampling to keep {num_to_keep} points...")
    
    # Generate random, unique indices to keep
    # replace=False ensures we don't pick the same point twice
    indices = np.random.choice(num_points, num_to_keep, replace=False)
    
    # Extract only the randomly selected points
    downsampled_data = vertex_data[indices]
    
    # Reconstruct the PLY element with the exact same data schema
    new_vertex_element = PlyElement.describe(downsampled_data, 'vertex')
    
    # Write the new PLY file (keeping it in binary or text based on the original)
    print(f"Saving to {output_file}...")
    PlyData([new_vertex_element], text=plydata.text).write(output_file)
    print("Done!")

# Example usage:
downsample_ply_numpy("/mnt/c/Unreal/Productions/init_pc/init_pc.ply", "/mnt/c/Unreal/Productions/init_pc/init_pc_downsampled.ply")