import numpy as np
from loguru import logger

class ConvergenceChecker:
    """Check convergence of FCM/FCLM algorithms."""
    
    @staticmethod
    def check(U_new, U_old, epsilon=1e-5):
        """Check if Frobenius norm change is below epsilon."""
        change = ConvergenceChecker.compute_change(U_new, U_old)
        return change < epsilon
    
    @staticmethod
    def compute_change(U_new, U_old):
        """Frobenius norm of difference."""
        return np.linalg.norm(U_new - U_old, 'fro')
    
    @staticmethod
    def check_objective(J_new, J_old, epsilon=1e-6):
        """Check if objective change is below epsilon."""
        return abs(J_new - J_old) < epsilon
    
    @staticmethod
    def early_stopping(J_history, patience=5, min_delta=1e-6):
        """Check if objective hasn't improved for `patience` iterations."""
        if len(J_history) < patience + 1:
            return False
        recent = J_history[-patience:]
        return max(recent) - min(recent) < min_delta
