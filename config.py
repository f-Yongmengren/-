import os

TRAIN_DIR=R'D:\pycharm project\digital_processing\digital_process\process_data\train'
TEST_DIR = R'D:\pycharm project\digital_processing\digital_process\process_data\test'
CLASS_NAMES=[
    'African people and villages',
    'Beach',
    'Historical buildings',
    'Buses',
    'Dinosaurs',
    'Elephants',
    'Flowers',
    'Horses',
    'Mountains and glaciers',
    'Food',
    'Dogs',
    'Lizards',
    'Fashion',
    'Sunsets',
    'Cars',
    'Waterfall',
    'Antiques',
    'Battle ship',
    'Skiing',
    'Desserts'
]


# 特征提取参数
GLCM_FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
SIFT_NUM_CLUSTERS = 100  # BOW聚类中心数量
COLOR_BINS = 32          # 颜色直方图bins
LBP_RADIUS = 3           # LBP半径
LBP_POINTS = 24          # LBP点数

# 模型参数
SVM_C = 1.0
KNN_N_NEIGHBORS = 5

# 输出路径
OUTPUT_DIR = './output'
RESULTS_CSV = os.path.join(OUTPUT_DIR, 'results.csv')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_CLASSES =20