import argparse
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from demo import DefectDetector
import utils.visualization as vis

def process_folder(input_folder, model_path, output_folder, save_ply=True):
    # Initialize detector
    detector = DefectDetector(model_path)
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .ply files
    input_files = list(Path(input_folder).glob('*.ply'))
    
    # Process each file
    for file_path in tqdm(input_files, desc="Processing files"):
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            
            # Predict
            labels = detector.predict(points)
            
            # Statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"\nResults for {file_path.name}:")
            for label, count in zip(unique_labels, counts):
                print(f"{detector.label_names[label]}: {count} points")
            
            # Save results
            if save_ply:
                colors = vis.create_label_colors(labels)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                output_file = output_path / f"result_{file_path.stem}.ply"
                o3d.io.write_point_cloud(str(output_file), pcd)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Batch Defect Detection')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing .ply files')
    parser.add_argument('--model', type=str, required=True, help='Model weights path')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    process_folder(args.input, args.model, args.output)

if __name__ == "__main__":
    main()