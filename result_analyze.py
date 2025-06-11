import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from textwrap import wrap

# 设置中文支持（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('output/feature_fusion_results.csv')

# 1. 整体准确率分布图
plt.figure(figsize=(12, 6))
sns.boxplot(x='classifier', y='accuracy', data=df)
plt.title('不同分类器的准确率分布')
plt.ylabel('准确率')
plt.xlabel('分类器')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('classifier_accuracy_distribution.png', dpi=300)
plt.show()

# 2. 特征组合与准确率的热力图
# 创建特征数量列
df['feature_count'] = df['features'].apply(lambda x: len(x.split(',')))

# 准备热力图数据
heatmap_data = df.pivot_table(index='features', columns='classifier', values='accuracy')

# 按特征数量排序
feature_counts = df.groupby('features')['feature_count'].first()
sorted_features = feature_counts.sort_values().index
heatmap_data = heatmap_data.loc[sorted_features]

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, cbar_kws={'label': '准确率'})
plt.title('不同特征组合与分类器的准确率热力图')
plt.xlabel('分类器')
plt.ylabel('特征组合')
plt.tight_layout()
plt.savefig('feature_classifier_heatmap.png', dpi=300)
plt.show()

# 3. 特征组合准确率对比图
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='features', y='accuracy', hue='classifier', data=df, palette='viridis')

# 添加数据标签
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=9)

plt.title('不同特征组合的准确率对比')
plt.xticks(rotation=45, ha='right')
plt.xlabel('特征组合')
plt.ylabel('准确率')
plt.legend(title='分类器', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_combination_accuracy.png', dpi=300)
plt.show()

# 4. 特征数量与准确率的关系
plt.figure(figsize=(12, 7))
sns.lineplot(x='feature_count', y='accuracy', hue='classifier',
             data=df, marker='o', markersize=8, linewidth=2.5)

# 添加平均值线
svm_mean = df[df['classifier'] == 'svm'].groupby('feature_count')['accuracy'].mean()
knn_mean = df[df['classifier'] == 'knn'].groupby('feature_count')['accuracy'].mean()

plt.axhline(y=svm_mean.mean(), color='blue', linestyle='--', alpha=0.5, label='SVM平均')
plt.axhline(y=knn_mean.mean(), color='orange', linestyle='--', alpha=0.5, label='KNN平均')

# 标注每个特征数量点的平均值
for count in sorted(df['feature_count'].unique()):
    svm_val = svm_mean[count]
    knn_val = knn_mean[count]
    plt.text(count, svm_val + 0.01, f'{svm_val:.3f}', ha='center', fontsize=9, color='blue')
    plt.text(count, knn_val + 0.01, f'{knn_val:.3f}', ha='center', fontsize=9, color='orange')

plt.title('特征数量与准确率的关系')
plt.xlabel('特征数量')
plt.ylabel('准确率')
plt.legend(title='分类器')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(sorted(df['feature_count'].unique()))
plt.tight_layout()
plt.savefig('feature_count_vs_accuracy.png', dpi=300)
plt.show()

# 5. 最佳特征组合分析
best_svm = df[df['classifier'] == 'svm'].sort_values('accuracy', ascending=False).head(5)
best_knn = df[df['classifier'] == 'knn'].sort_values('accuracy', ascending=False).head(5)

# 合并最佳结果
best_results = pd.concat([best_svm, best_knn])

plt.figure(figsize=(14, 8))
ax = sns.barplot(x='features', y='accuracy', hue='classifier',
                 data=best_results, palette='coolwarm', dodge=True)

# 添加数据标签
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=10)

plt.title('最佳特征组合性能比较')
plt.xlabel('特征组合')
plt.ylabel('准确率')
plt.ylim(0.6, 0.85)
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('best_feature_combinations.png', dpi=300)
plt.show()

# 6. 各特征单独表现与组合效果对比
# 提取单个特征的表现
single_features = ['glcm', 'sift', 'lbp', 'color', 'cnn']
single_results = df[df['features'].isin(single_features)]

# 提取组合特征的表现
combined_results = df[~df['features'].isin(single_features)]

plt.figure(figsize=(14, 8))

# 创建子图
plt.subplot(1, 2, 1)
sns.barplot(x='features', y='accuracy', hue='classifier',
            data=single_results, palette='Set2')
plt.title('单个特征性能')
plt.xlabel('特征')
plt.ylabel('准确率')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.barplot(x='features', y='accuracy', hue='classifier',
            data=combined_results, palette='Set1')
plt.title('特征组合性能')
plt.xlabel('特征组合')
plt.ylabel('准确率')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.suptitle('单个特征与特征组合性能对比', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.savefig('single_vs_combined_features.png', dpi=300)
plt.show()

# 7. SVM与KNN性能差异分析
df['accuracy_diff'] = df.groupby('features')['accuracy'].transform(lambda x: x.max() - x.min())

plt.figure(figsize=(14, 8))
sns.barplot(x='features', y='accuracy_diff',
            data=df.drop_duplicates('features'),
            palette='rocket')
plt.title('SVM与KNN准确率差异')
plt.xlabel('特征组合')
plt.ylabel('准确率差异 (SVM - KNN)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加平均差异线
mean_diff = df['accuracy_diff'].mean()
plt.axhline(y=mean_diff, color='red', linestyle='--', label=f'平均差异: {mean_diff:.3f}')
plt.legend()

plt.tight_layout()
plt.savefig('svm_knn_accuracy_difference.png', dpi=300)
plt.show()