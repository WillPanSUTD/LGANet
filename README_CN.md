# 密封钉缺陷检测系统

## 项目简介

本项目是一个基于图神经网络的密封钉缺陷检测系统，使用点云数据进行缺陷分类。系统采用自定义的图注意力机制进行特征提取，可以有效识别密封钉表面的多种缺陷类型。

### 支持的缺陷类型

- 破裂 (Burst)
- 凹坑 (Pit)
- 污渍 (Stain)
- 翘曲 (Warpage)
- 针孔 (Pinhole)
- 其他背景类别

## 技术架构

- 基于PyTorch深度学习框架
- 使用图注意力网络(GAT)进行特征提取
- 自定义EdgeConv和GraphConv模块处理点云数据
- 多尺度特征融合的编码器-解码器结构

## 环境要求

- Python 3.x
- PyTorch
- NumPy
- CUDA（推荐用于GPU加速）
- 自定义的pointops库（用于点云操作）

## 环境配置

### 使用 Conda 配置环境

1. 创建新的 Conda 环境：
```bash
conda env create -f environment.yml
```

2. 激活新环境：
conda activate sealingnail

3. 安装必要的包：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

4. 安装其他依赖：
conda install numpy scipy scikit-learn
conda install -c conda-forge open3d
conda install tqdm tensorboard h5py


sudo apt install g++
sudo apt install build-essential
sudo apt install ninja-build
ninja --version

5. 安装自定义的pointops库：
cd lib/pointops
python setup.py install

6. 验证安装：
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

## 数据集结构

数据集组织方式：
```
data/sealingNail_normal/
    ├── train/        # 训练数据集
    └── test/         # 测试数据集
```

每个点云数据包含以下特征：
- 3D坐标 (x, y, z)
- 表面特征 (3维)
- 类别标签

## 模型架构

模型采用编码器-解码器结构：
1. 编码器：多层图注意力模块，逐层下采样提取特征
2. 解码器：通过跳跃连接融合多尺度特征
3. 分类头：最终输出每个点的类别预测

## 使用方法

### 训练模型

```bash
python train.py
```


### 演示效果

```bash
python demo.py --input data/test_sample.ply --model checkpoints/best_model.pth --output results/ --vis --save_ply
```

单个
python demo.py --input data/test_sample.ply --model checkpoints/best_model.pth --output results/

批量检测
python batch_demo.py --input data/test_folder/ --model checkpoints/best_model.pth --output results/

可视化
python visualize.py --ply results/labeled_cloud.ply

导出html报告
python export_report.py --input results/ --output reports/

训练过程会自动：
- 加载并预处理数据集
- 计算类别权重以处理数据不平衡
- 定期保存模型检查点
- 输出训练和验证指标

### 模型评估

系统使用多个评估指标：
- 总体准确率 (OA)
- 平均准确率 (mAcc)
- 平均交并比 (mIoU)
- 各类别IoU

## 项目结构

```
├── model/          # 模型定义
│   └── sem/        # 语义分割模型
├── util/           # 工具函数
├── lib/            # 自定义算子库
│   └── pointops/   # 点云操作
├── data/           # 数据集
└── train.py        # 训练脚本
```

## 注意事项

1. 确保CUDA环境正确配置
建议您先检查系统的 CUDA 版本：
```bash
nvidia-smi
2. 首次运行前需编译pointops库
3. 根据实际GPU内存大小调整batch_size
4. 训练日志和模型权重保存在logs目录下
