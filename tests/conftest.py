import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X.astype(np.float64), y


@pytest.fixture(scope="session")
def iris_X(iris_data):
    return iris_data[0]


@pytest.fixture(scope="session")
def iris_y(iris_data):
    return iris_data[1]


@pytest.fixture(scope="session")
def small_blobs():
    X, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42, cluster_std=0.8)
    return X.astype(np.float64), y


@pytest.fixture(scope="session")
def blobs_X(small_blobs):
    return small_blobs[0]


@pytest.fixture(scope="session")
def blobs_y(small_blobs):
    return small_blobs[1]


@pytest.fixture(scope="function")
def tiny_data():
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]], dtype=np.float64)
    return X


@pytest.fixture(scope="function")
def sample_weblog_df():
    import pandas as pd
    return pd.DataFrame({
        "ip": ["192.168.1.1", "192.168.1.1", "192.168.1.2", "192.168.1.1", "192.168.1.3"],
        "url": ["/page1", "/page2", "/page1", "/page3", "/page2"],
        "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:05:00",
                                      "2024-01-01 10:10:00", "2024-01-01 11:00:00",
                                      "2024-01-01 10:15:00"]),
        "method": ["GET", "GET", "POST", "GET", "GET"],
        "status": [200, 200, 200, 200, 200],
        "user_agent": ["Mozilla/5.0", "Mozilla/5.0", "python-requests/2.28",
                       "Mozilla/5.0", "curl/7.68"],
    })
