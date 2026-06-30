"""
XAI Module for ADE-FCM: Explainable AI tools for clustering interpretation.
Provides cluster summaries, feature importance, SHAP-based explanations,
NL descriptions, and visualization utilities.
"""

from .cluster_explainer import ClusterExplainer
from .shap_explainer import ShapExplainer
from .visualizer import XAIVisualizer

__all__ = [
    "ClusterExplainer",
    "ShapExplainer",
    "XAIVisualizer",
]
