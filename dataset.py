# dataset.py
import os
import numpy as np
from PIL import Image
import config


class MyDataset:
    def __init__(self, type="train", transform=None):
        """
        修改后的数据集类，符合项目要求

        参数:
            type: 'train' 或 'test'
            transform: 可选的图像变换函数
        """
        self.type = type
        self.transform = transform
        self.data_dir = config.TRAIN_DIR if self.type == 'train' else config.TEST_DIR
        self.label_list = sorted(os.listdir(self.data_dir))  # 确保标签顺序一致

        # 创建图像路径和标签的映射
        self.image_paths = []
        self.labels = []
        self.image_names = []

        # 遍历所有类别目录
        for label_idx, label_name in enumerate(self.label_list):
            label_dir = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            # 获取当前类别下的所有图像
            img_files = sorted(os.listdir(label_dir))
            for img_name in img_files:
                img_path = os.path.join(label_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label_idx)  # 使用数字标签
                self.image_names.append(img_name)

    def __getitem__(self, index):
        """获取单个样本"""
        img_path = self.image_paths[index]
        label = self.labels[index]
        img_name = self.image_names[index]

        # 加载图像
        pil_img = Image.open(img_path).convert('RGB')

        # 应用变换（如果有）
        if self.transform:
            pil_img = self.transform(pil_img)

        # 转换为数组图像
        array_img = np.array(pil_img)

        return {
            'pil_image': pil_img,
            'array_image': array_img,
            'label': label,
            'image_name': img_name
        }

    def __len__(self):
        return len(self.image_paths)

    def get_images_array(self):
        """获取所有图像数组"""
        images = []
        for i in range(len(self)):
            item = self[i]
            images.append(item['array_image'])
        return images

    def get_labels(self):
        """获取所有标签"""
        return self.labels

    def get_image_names(self):
        """获取所有图像名称"""
        return self.image_names