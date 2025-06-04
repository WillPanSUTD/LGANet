import numpy as np

def create_label_colors(labels):
    """为不同类别的标签创建颜色映射
    
    Args:
        labels: 标签数组，每个点的类别标签
        
    Returns:
        colors: 与点云数量相同的RGB颜色数组
    """
    # 定义颜色映射，每个类别对应一个RGB颜色
    color_map = {
        0: [0.7, 0.7, 0.7],  # Background - 灰色
        1: [1.0, 0.0, 0.0],  # Burst - 红色
        2: [0.0, 1.0, 0.0],  # Pit - 绿色
        3: [0.0, 0.0, 1.0],  # Stain - 蓝色
        4: [1.0, 1.0, 0.0],  # Warpage - 黄色
        5: [1.0, 0.0, 1.0],  # Pinhole - 紫色
    }
    
    # 为每个点分配颜色
    colors = np.zeros((len(labels), 3))
    for label in np.unique(labels):
        mask = labels == label
        colors[mask] = color_map.get(label, [0.5, 0.5, 0.5])  # 默认颜色为灰色
    
    return colors