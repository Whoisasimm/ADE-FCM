import time
import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, rand_score,
    f1_score, accuracy_score
)
from loguru import logger

class Evaluator:
    """Comprehensive clustering evaluation."""
    
    @staticmethod
    def silhouette_score(X, labels):
        if len(set(labels)) > 1:
            return silhouette_score(X, labels)
        return 0.0
    
    @staticmethod
    def davies_bouldin_index(X, labels):
        if len(set(labels)) > 1:
            return davies_bouldin_score(X, labels)
        return float('inf')
    
    @staticmethod
    def calinski_harabasz_score(X, labels):
        if len(set(labels)) > 1:
            return calinski_harabasz_score(X, labels)
        return 0.0
    
    @staticmethod
    def adjusted_rand_index(y_true, y_pred):
        return adjusted_rand_score(y_true, y_pred)
    
    @staticmethod
    def normalized_mutual_info(y_true, y_pred):
        return normalized_mutual_info_score(y_true, y_pred)
    
    @staticmethod
    def rand_index(y_true, y_pred):
        return rand_score(y_true, y_pred)
    
    @staticmethod
    def compute_all(X, labels, y_true=None):
        metrics = {
            'silhouette': Evaluator.silhouette_score(X, labels),
            'davies_bouldin': Evaluator.davies_bouldin_index(X, labels),
            'calinski_harabasz': Evaluator.calinski_harabasz_score(X, labels),
        }
        if y_true is not None:
            metrics.update({
                'adjusted_rand_index': Evaluator.adjusted_rand_index(y_true, labels),
                'normalized_mutual_info': Evaluator.normalized_mutual_info(y_true, labels),
                'rand_index': Evaluator.rand_index(y_true, labels),
            })
        return metrics
    
    @staticmethod
    def execution_time(func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
