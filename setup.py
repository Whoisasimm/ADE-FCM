from setuptools import setup, find_packages

setup(
    name="ade-fcm",
    version="1.0.0",
    description="ADE-FCM: An Adaptive Explainable Fuzzy C-Means Framework for Automated Clustering",
    author="Khalil Benihya, Ahmed Khaled",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "spark": ["pyspark>=3.3.0"],
        "gpu": ["cupy-cuda11x>=11.0.0"],
        "streaming": ["confluent-kafka>=1.9.0"],
        "xai": ["shap>=0.40.0"],
        "mlflow": ["mlflow>=1.25.0"],
        "all": ["pyspark", "cupy-cuda11x", "confluent-kafka", "shap", "mlflow"],
    },
    entry_points={
        "console_scripts": [
            "ade-fcm=src.main:main",
            "ade-fcm-benchmark=benchmarks.main:main",
            "ade-fcm-stream=streaming.main:main",
            "ade-fcm-xai=xai.main:main",
            "ade-fcm-ablation=ablation.main:main",
        ],
    },
    python_requires=">=3.8",
)
