# feature_extractor.py
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from digital_process.utils import (
    extract_glcm_features, extract_sift_features, extract_lbp_features,
    extract_color_histogram, train_bow_model, extract_bow_features,
    parallel_feature_extraction
)
from digital_process.config import SIFT_NUM_CLUSTERS
import joblib
import os
from config import OUTPUT_DIR


class FeatureExtractor:
    def __init__(self):
        self.bow_model = None
        self.cnn_model = self._build_cnn_model()
        self.transform = self._build_transform()

    def _build_cnn_model(self):
        """构建PyTorch CNN特征提取模型"""
        # 使用预训练的MobileNetV2
        model = models.mobilenet_v2(pretrained=True)

        # 移除最后的分类层，保留特征提取部分
        model = torch.nn.Sequential(*list(model.children())[:-1])

        # 设置为评估模式
        model.eval()

        return model

    def _build_transform(self):
        """构建图像预处理转换"""
        return transforms.Compose([
            transforms.Resize(256),  # 调整大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                                 std=[0.229, 0.224, 0.225])
        ])

    def train_bow(self, images):
        """训练BOW模型"""
        # 提取所有图像的SIFT描述子
        all_descriptors = []
        for img in images:
            descriptors = extract_sift_features(img)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)

        if not all_descriptors:
            raise ValueError("No SIFT descriptors found in training images")

        # 训练BOW模型
        self.bow_model = train_bow_model(all_descriptors, SIFT_NUM_CLUSTERS)

        # 保存BOW模型
        bow_path = os.path.join(OUTPUT_DIR, 'bow_model.joblib')
        joblib.dump(self.bow_model, bow_path)
        print(f"BOW model trained and saved to {bow_path}")

    def extract_features(self, images, feature_types=['glcm', 'sift', 'lbp', 'color', 'cnn']):
        """
        提取多种特征并融合

        参数:
            images: 图像列表 (numpy数组)
            feature_types: 要提取的特征类型列表

        返回:
            融合后的特征矩阵 (n_samples, n_features)
        """
        all_features = []
        feature_names = []

        # 提取各种特征
        if 'glcm' in feature_types:
            glcm_feats = parallel_feature_extraction(images, extract_glcm_features)
            all_features.append(glcm_feats)
            feature_names.append('GLCM')
            print(f"Extracted GLCM features: {glcm_feats.shape}")

        if 'sift' in feature_types:
            if self.bow_model is None:
                raise RuntimeError("BOW model not trained. Call train_bow() first.")

            # 提取SIFT特征并使用BOW编码
            sift_feats = []
            for img in images:
                descriptors = extract_sift_features(img)
                bow_feat = extract_bow_features(descriptors, self.bow_model)
                sift_feats.append(bow_feat)

            sift_feats = np.array(sift_feats)
            all_features.append(sift_feats)
            feature_names.append('SIFT-BOW')
            print(f"Extracted SIFT-BOW features: {sift_feats.shape}")

        if 'lbp' in feature_types:
            lbp_feats = parallel_feature_extraction(images, extract_lbp_features)
            all_features.append(lbp_feats)
            feature_names.append('LBP')
            print(f"Extracted LBP features: {lbp_feats.shape}")

        if 'color' in feature_types:
            color_feats = parallel_feature_extraction(images, extract_color_histogram)
            all_features.append(color_feats)
            feature_names.append('Color')
            print(f"Extracted Color features: {color_feats.shape}")

        if 'cnn' in feature_types:
            cnn_feats = []
            for img in images:
                features = self.extract_cnn_features(img)
                cnn_feats.append(features)

            cnn_feats = np.array(cnn_feats)
            all_features.append(cnn_feats)
            feature_names.append('CNN')
            print(f"Extracted CNN features: {cnn_feats.shape}")

        # 合并所有特征
        fused_features = np.hstack(all_features)
        print(f"Feature types: {feature_names}")
        print(f"Fused features shape: {fused_features.shape}")

        return fused_features

    def extract_cnn_features(self, image):
        """使用PyTorch提取CNN特征"""
        # 转换图像格式
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 灰度图转换为RGB
                image = np.stack((image,) * 3, axis=-1)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # 应用预处理
        input_tensor = self.transform(pil_image)

        # 添加批处理维度
        input_batch = input_tensor.unsqueeze(0)

        # 使用GPU如果可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model.to(device)
        input_batch = input_batch.to(device)

        # 提取特征（不计算梯度）
        with torch.no_grad():
            features = self.cnn_model(input_batch)

        # 转换为numpy数组并展平
        features = features.squeeze(0).cpu().numpy().flatten()

        return features