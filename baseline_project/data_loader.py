"""
Data loader for weblog and synthetic datasets.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer

class DataLoader:
    """Loads data from various sources."""
    
    def __init__(self):
        self.data = None
        
    def load_weblog_data(self, filepath):
        """Load weblog data from CSV/TSV."""
        logger.info(f"Loading weblog data from {filepath}")
        ext = Path(filepath).suffix
        if ext == '.gz' or ext == '.csv':
            self.data = pd.read_csv(filepath, compression='gzip' if ext == '.gz' else None)
        elif ext == '.tsv':
            self.data = pd.read_csv(filepath, sep='\t')
        elif ext == '.json':
            self.data = pd.read_json(filepath)
        else:
            self.data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data
    
    def load_synthetic_data(self, n_samples=1000, n_features=10, n_clusters=5, random_state=42, noise=0.05):
        """Generate synthetic clustered data."""
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                          centers=n_clusters, random_state=random_state, cluster_std=noise)
        logger.info(f"Generated synthetic data: {X.shape}, {n_clusters} clusters")
        return X, y
    
    def load_benchmark_dataset(self, name='iris'):
        """Load benchmark datasets: iris, wine, digits, breast_cancer."""
        datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'digits': load_digits,
            'breast_cancer': load_breast_cancer
        }
        if name not in datasets:
            raise ValueError(f"Unknown dataset {name}. Choose from {list(datasets.keys())}")
        data = datasets[name]()
        X, y = data.data, data.target
        logger.info(f"Loaded {name} dataset: {X.shape}")
        return X, y
