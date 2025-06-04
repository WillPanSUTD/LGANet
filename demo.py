import argparse
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from model.sem.GraphAttention import graphAttention_seg_repro
import util.visualization as vis

class DefectDetector:
    def __init__(self, model_path, feat_dim=6, num_class=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = graphAttention_seg_repro(c=feat_dim, k=num_class).to(self.device)
        self.load_model(model_path)
        self.label_names = ['Background', 'Burst', 'Pit', 'Stain', 'Warpage', 'Pinhole', 'Other1', 'Other2']
        
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"模型加载成功：{model_path}")
        except Exception as e:
            print(f"模型加载错误：{str(e)}")
            raise
        
    def predict(self, points):
        with torch.no_grad():
            # 准备输入数据
            coords = points[:, :3]  # 坐标
            normals = points[:, 3:6] if points.shape[1] > 3 else torch.zeros_like(coords)  # 法向量
            
            # 转换为张量并移动到设备
            coords = torch.FloatTensor(coords).to(self.device)
            normals = torch.FloatTensor(normals).to(self.device)
            offset = torch.tensor([coords.size(0)], dtype=torch.int32, device=self.device)
            
            # 模型推理
            pred = self.model([coords, normals, offset])
            pred = torch.argmax(pred, dim=1)
            
        return pred.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='密封钉缺陷检测演示程序')
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--output', type=str, default='results', help='结果保存目录')
    parser.add_argument('--vis', action='store_true', help='是否显示可视化结果')
    parser.add_argument('--save_ply', action='store_true', help='是否保存结果点云文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载点云数据
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    if not np.asarray(pcd.normals).size:
        pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    points = np.concatenate([points, normals], axis=1)
    
    # 初始化检测器并进行预测
    detector = DefectDetector(args.model)
    labels = detector.predict(points)
    
    # 统计各类缺陷数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n检测结果统计：")
    for label, count in zip(unique_labels, counts):
        print(f"{detector.label_names[label]}: {count} 点")
    
    # 可视化结果
    if args.vis or args.save_ply:
        colors = vis.create_label_colors(labels)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if args.save_ply:
            o3d.io.write_point_cloud(str(output_path / "result.ply"), pcd)
            print(f"结果已保存至: {output_path / 'result.ply'}")
            
        if args.vis:
            o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()