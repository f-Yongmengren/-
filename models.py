# models.py
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
from config import OUTPUT_DIR


class Classifier:
    def __init__(self, classifier_type='svm', **kwargs):
        self.classifier_type = classifier_type
        self.model = None
        self.scaler = StandardScaler()

        if classifier_type == 'svm':
            self.model = SVC(C=kwargs.get('C', 1.0), kernel='rbf', probability=True)
        elif classifier_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=kwargs.get('n_neighbors', 5))
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.model)
        ])

    def train(self, X, y):
        """训练分类器"""
        self.pipeline.fit(X, y)

    def predict(self, X):
        """预测类别"""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.pipeline.predict_proba(X)

    def save(self, filename):
        """保存模型"""
        path = os.path.join(OUTPUT_DIR, filename)
        dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load(self, filename):
        """加载模型"""
        path = os.path.join(OUTPUT_DIR, filename)
        self.pipeline = load(path)
        print(f"Model loaded from {path}")
        return self