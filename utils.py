# utils.py
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import os
from config import GLCM_FEATURES, SIFT_NUM_CLUSTERS, COLOR_BINS, LBP_RADIUS, LBP_POINTS


def extract_glcm_features(image):
    """提取灰度共生矩阵(GLCM)特征"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 将图像转换为8位灰度
    gray = (gray * 255).astype(np.uint8)

    # 计算灰度共生矩阵
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)

    features = []
    for prop in GLCM_FEATURES:
        feature = graycoprops(glcm, prop)
        features.append(feature.mean())

    return np.array(features)


def extract_sift_features(image):
    """提取SIFT特征点"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.array([])


def extract_lbp_features(image):
    """提取局部二值模式(LBP)纹理特征"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # 归一化
    return hist


def extract_color_histogram(image):
    """提取颜色直方图特征"""
    if len(image.shape) == 2:  # 如果是灰度图
        hist = cv2.calcHist([image], [0], None, [COLOR_BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 计算每个通道的直方图
    h_hist = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [COLOR_BINS], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [COLOR_BINS], [0, 256])

    # 归一化并拼接
    hist = np.concatenate([
        cv2.normalize(h_hist, h_hist).flatten(),
        cv2.normalize(s_hist, s_hist).flatten(),
        cv2.normalize(v_hist, v_hist).flatten()
    ])
    return hist


def train_bow_model(descriptors_list, num_clusters=SIFT_NUM_CLUSTERS):
    """训练BOW模型"""
    # 合并所有描述子
    all_descriptors = np.vstack(descriptors_list)

    # 使用MiniBatchKMeans进行聚类
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
    kmeans.fit(all_descriptors)
    return kmeans


def extract_bow_features(descriptors, bow_model):
    """使用BOW模型提取特征"""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(bow_model.n_clusters)

    # 预测每个描述子所属的簇
    labels = bow_model.predict(descriptors)

    # 计算直方图
    hist, _ = np.histogram(labels, bins=range(bow_model.n_clusters + 1))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # 归一化
    return hist


def extract_cnn_features(image, model):
    """使用预训练的CNN模型提取特征"""
    # 预处理图像
    img = cv2.resize(image, (224, 224))
    img = img.astype('float32') / 255.0
    if len(img.shape) == 2:  # 如果是灰度图，转换为3通道
        img = np.stack((img,) * 3, axis=-1)
    img = np.expand_dims(img, axis=0)

    # 提取特征
    features = model.predict(img, verbose=0)
    return features.flatten()


def parallel_feature_extraction(images, feature_fn, **kwargs):
    """并行特征提取"""
    features = Parallel(n_jobs=-1)(
        delayed(feature_fn)(img, **kwargs) for img in images
    )
    return np.array(features)