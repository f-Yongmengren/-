# train.py
import numpy as np
import pandas as pd
import os
import time
from dataset import MyDataset
from feature_extractor import FeatureExtractor
from models import Classifier
from config import RESULTS_CSV, OUTPUT_DIR


def train_and_evaluate(feature_types, classifier_type='svm'):
    """
    训练和评估模型

    参数:
        feature_types: 要使用的特征类型列表 (['glcm', 'sift', 'lbp', 'color', 'cnn'])
        classifier_type: 分类器类型 ('svm' 或 'knn')
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据集
    print("Loading datasets...")
    train_dataset = MyDataset('train')
    test_dataset = MyDataset('test')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 准备训练和测试数据
    train_images = train_dataset.get_images_array()
    train_labels = train_dataset.get_labels()
    test_images = test_dataset.get_images_array()
    test_labels = test_dataset.get_labels()
    test_names = test_dataset.get_image_names()

    # 特征提取器
    feature_extractor = FeatureExtractor()

    # 训练BOW模型（如果使用SIFT特征）
    if 'sift' in feature_types:
        print("Training BOW model...")
        feature_extractor.train_bow(train_images)

    # 提取训练集特征
    print("Extracting training features...")
    train_features = feature_extractor.extract_features(train_images, feature_types)

    # 提取测试集特征
    print("Extracting test features...")
    test_features = feature_extractor.extract_features(test_images, feature_types)

    # 训练分类器
    print(f"Training {classifier_type.upper()} classifier...")
    classifier = Classifier(classifier_type)
    start_time = time.time()
    classifier.train(train_features, train_labels)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # 保存模型
    model_name = f"{'_'.join(feature_types)}_{classifier_type}.joblib"
    classifier.save(model_name)

    # 在测试集上评估
    print("Evaluating on test set...")
    start_time = time.time()
    pred_labels = classifier.predict(test_features)
    test_time = time.time() - start_time
    print(f"Testing completed in {test_time:.2f} seconds")

    # 计算准确率
    accuracy = np.mean(pred_labels == test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 保存结果到CSV
    results = []
    for i in range(len(test_labels)):
        results.append({
            'image_id': test_names[i],
            'true_label': test_labels[i],
            'pred_label': pred_labels[i]
        })

    # df = pd.DataFrame(results)
    # df.to_csv(RESULTS_CSV, index=False)
    # print(f"Results saved to {RESULTS_CSV}")
    result_filename = f"results_{'_'.join(feature_types)}_{classifier_type}.csv"
    result_path = os.path.join(OUTPUT_DIR, result_filename)

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    return accuracy

    return accuracy


def feature_fusion_experiment():
    """执行不同特征融合的实验"""
    feature_combinations = [
        ['glcm'],
        ['sift'],
        ['lbp'],
        ['color'],
        ['cnn'],  # 使用PyTorch实现的CNN特征
        ['glcm', 'sift'],
        ['glcm', 'lbp'],
        ['glcm', 'color'],
        ['glcm', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['sift', 'lbp'],
        ['sift', 'color'],
        ['sift', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['lbp', 'color'],
        ['lbp', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['color', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['glcm', 'sift', 'lbp'],
        ['glcm', 'sift', 'color'],
        ['glcm', 'sift', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['sift', 'lbp', 'color'],
        ['sift', 'lbp', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['glcm', 'sift', 'lbp', 'color'],
        ['glcm', 'sift', 'lbp', 'cnn'],  # 使用PyTorch实现的CNN特征
        ['glcm', 'sift', 'lbp', 'color', 'cnn']  # 使用PyTorch实现的CNN特征
    ]

    results = []

    for features in feature_combinations:
        for classifier in ['svm', 'knn']:
            print(f"\n{'=' * 50}")
            print(f"Experiment: Features={features}, Classifier={classifier}")
            print(f"{'=' * 50}")

            try:
                acc = train_and_evaluate(features, classifier)
                results.append({
                    'features': ', '.join(features),
                    'classifier': classifier,
                    'accuracy': acc
                })
            except Exception as e:
                print(f"Experiment failed: {str(e)}")
                results.append({
                    'features': ', '.join(features),
                    'classifier': classifier,
                    'accuracy': -1
                })

    # 保存实验结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'feature_fusion_results.csv'), index=False)
    print("All experiments completed. Results saved.")


if __name__ == "__main__":
    # 执行所有特征融合实验
    feature_fusion_experiment()