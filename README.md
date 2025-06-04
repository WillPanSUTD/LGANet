
LGANet： An improved Graph Attention Network for Semantic Segmentation of Industrial Point Clouds in Automotive Battery Sealing Nail Defect Detection


# Sealing Nail Defect Detection System

## Project Overview

This project implements a defect detection system for automotive battery sealing nails based on graph neural networks (GNNs). It uses point cloud data to classify surface defects. The system leverages a custom graph attention mechanism for feature extraction, enabling robust detection of various defect types on metal surfaces.

### Supported Defect Types

* Burst
* Pit
* Stain
* Warpage
* Pinhole
* Other background categories

---

## Technical Architecture

* Built on the PyTorch deep learning framework
* Employs Graph Attention Networks (GATs) for feature extraction
* Custom `EdgeConv` and `GraphConv` modules for point cloud processing
* Multi-scale encoder–decoder structure for feature fusion

---

## Requirements

* Python 3.x
* PyTorch
* NumPy
* CUDA (recommended for GPU acceleration)
* Custom `pointops` library for point cloud operations

---

## Environment Setup

### Recommended: Use Conda

1. Create a new Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate sealingnail
   ```

3. Install PyTorch with CUDA:

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. Install other dependencies:

   ```bash
   conda install numpy scipy scikit-learn
   conda install -c conda-forge open3d
   conda install tqdm tensorboard h5py
   ```

5. Ensure C++/build tools (Linux):

   ```bash
   sudo apt install g++
   sudo apt install build-essential
   sudo apt install ninja-build
   ninja --version
   ```

6. Install custom pointops library:

   ```bash
   cd lib/pointops
   python setup.py install
   ```

7. Verify installation:

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

---

## Dataset Structure
Data can be downloaded from 
TXT format: https://drive.google.com/file/d/1zyCQI2J5TFcRKKLp0KkUvsXCY-D-nVCV/view?usp=drive_link
NPZ format: https://drive.google.com/file/d/1_ni0_XzQVsQ5ehoDo1MfV0wjTkiN1XjO/view?usp=drive_link
The dataset is organized as follows:

```
data/sealingNail_normal/
├── train/    # Training set
└── test/     # Test set
```

Each point cloud file contains:

* 3D coordinates (x, y, z)
* Surface features (3 channels)
* Semantic labels

---

## Model Architecture

The model adopts an encoder–decoder structure:

1. **Encoder**: Multi-layer graph attention modules with progressive downsampling
2. **Decoder**: Multi-scale feature fusion via skip connections
3. **Classification Head**: Outputs per-point semantic predictions

---

## Usage

### Train the Model

```bash
python train.py
```

Training includes:

* Data loading and preprocessing
* Class weight computation to handle imbalance
* Model checkpoint saving
* Real-time training and validation metrics

---

### Demo Inference

Run on a single sample:

```bash
python demo.py --input data/test_sample.ply --model checkpoints/best_model.pth --output results/
```

Batch inference:

```bash
python batch_demo.py --input data/test_folder/ --model checkpoints/best_model.pth --output results/
```

Visualization:

```bash
python visualize.py --ply results/labeled_cloud.ply
```

Export HTML report:

```bash
python export_report.py --input results/ --output reports/
```

---

## Model Evaluation

The system supports multiple evaluation metrics:

* Overall Accuracy (OA)
* Mean Accuracy (mAcc)
* Mean Intersection over Union (mIoU)
* Per-class IoU

---

## Project Structure

```
├── model/           # Model definitions
│   └── sem/         # Semantic segmentation modules
├── util/            # Utility functions
├── lib/             # Custom operators
│   └── pointops/    # Point cloud operations
├── data/            # Datasets
└── train.py         # Training script
```

---

## Notes

1. Ensure your CUDA environment is correctly installed
   Check with:

   ```bash
   nvidia-smi
   ```

2. Compile the pointops library before first run

3. Adjust `batch_size` according to available GPU memory

4. Training logs and model weights are saved under the `logs/` directory


