import os
import numpy as np
from tqdm import tqdm

def convert_to_npz(src_dir, dst_dir, split):
    """Convert txt point cloud data to npz format
    Args:
        src_dir: source directory containing txt files
        dst_dir: destination directory for npz files
        split: 'train' or 'test'
    """
    src_path = os.path.join(src_dir, split)
    dst_path = os.path.join(dst_dir, split)
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    files = os.listdir(src_path)
    for file in tqdm(files, desc=f'Converting {split} set'):
        txt_path = os.path.join(src_path, file)
        npz_path = os.path.join(dst_path, os.path.splitext(file)[0] + '.npz')
        
        # Load txt data
        points = np.loadtxt(txt_path, dtype=np.float32)
        
        # Save as npz
        np.savez_compressed(npz_path, points=points)

def main():
    src_root = './data/sealingNail_normal'
    dst_root = './data/sealingNail_npz'
    
    # Convert both train and test sets
    convert_to_npz(src_root, dst_root, 'train')
    convert_to_npz(src_root, dst_root, 'test')
    
    print('Dataset conversion completed!')

if __name__ == '__main__':
    main()