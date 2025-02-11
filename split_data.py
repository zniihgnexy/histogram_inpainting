import os
import shutil
import random

def split_dataset(data_root, output_root, train_ratio=0.7):
    """
    按照 70% 训练集，30% 验证集拆分数据集。
    
    :param data_root: 原始数据集路径 (e.g., "raw_data/diffusion_edge/data")
    :param output_root: 拆分后的数据存放路径 (e.g., "output/diffusion_edge")
    :param train_ratio: 训练集占比，默认为 0.7
    """
    image_dir = os.path.join(data_root, "image/raw")
    edge_dir = os.path.join(data_root, "edge/raw")

    train_image_dir = os.path.join(output_root, "train/image")
    train_edge_dir = os.path.join(output_root, "train/edge")
    val_image_dir = os.path.join(output_root, "val/image")
    val_edge_dir = os.path.join(output_root, "val/edge")

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_edge_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_edge_dir, exist_ok=True)

    images = sorted(os.listdir(image_dir))
    edges = sorted(os.listdir(edge_dir))
    assert len(images) == len(edges), "图像和边缘标签数量必须匹配！"

    data_pairs = list(zip(images, edges))
    random.shuffle(data_pairs)

    train_size = int(len(data_pairs) * train_ratio)
    train_pairs = data_pairs[:train_size]
    val_pairs = data_pairs[train_size:]

    print(f"总数据量: {len(data_pairs)}，训练集: {len(train_pairs)}，验证集: {len(val_pairs)}")

    # 复制训练数据
    for img, edge in train_pairs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
        shutil.copy(os.path.join(edge_dir, edge), os.path.join(train_edge_dir, edge))

    # 复制验证数据
    for img, edge in val_pairs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(val_image_dir, img))
        shutil.copy(os.path.join(edge_dir, edge), os.path.join(val_edge_dir, edge))

    print(f"数据集拆分完成！训练集存储于 {train_image_dir}，验证集存储于 {val_image_dir}")

# 运行数据拆分
split_dataset("raw_data/diffusion_edge/data", "output/diffusion_edge")
