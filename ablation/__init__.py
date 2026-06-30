"""
Ablation Study Module for ADE-FCM.
Systematically measures the impact of each novel component
by removing one component at a time and evaluating degradation.
"""

from .ablation_study import AblationStudy

__all__ = [
    "AblationStudy",
]
