import numpy as np
from loguru import logger

class ObjectiveFunction:
    """Compute objective functions and validity indices."""
    
    @staticmethod
    def compute_fcm(X, U, centers, m=2.0):
        """FCM cost function: J = sum_i sum_j u_ij^m * d(x_i, v_j)"""
        n, c = U.shape
        J = 0.0
        for j in range(c):
            diff = X - centers[j]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            J += np.sum(U[:, j]**m * dist)
        return J
    
    @staticmethod
    def compute_fclm(X, U, centers, m=2.0):
        """FCLM cost function (same formulation)."""
        return ObjectiveFunction.compute_fcm(X, U, centers, m)
    
    @staticmethod
    def compute_partition_coefficient(U):
        """PC = 1/n * sum_j sum_i u_ij^2"""
        n = U.shape[0]
        return np.sum(U**2) / n
    
    @staticmethod
    def compute_partition_entropy(U):
        """PEC = -1/n * sum_j sum_i u_ij * log(u_ij)"""
        n = U.shape[0]
        U = np.maximum(U, 1e-10)
        return -np.sum(U * np.log(U)) / n
    
    @staticmethod
    def compute_sse(X, labels, centers):
        """Sum of squared errors."""
        sse = 0.0
        for j, center in enumerate(centers):
            mask = labels == j
            if mask.any():
                sse += np.sum((X[mask] - center)**2)
        return sse
