from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vehiclepm",
    version="0.1.0",
    author="Kushal Khemani",
    author_email="kushal.khemani@gmail.com",
    description=(
        "Predictive maintenance for connected vehicles with V2X contextual "
        "data fusion, SHAP interpretability, and noise sensitivity analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kushalk0677/AI-Driven-Predictive-Maintenance-with-Real-Time-Contextual-Data-Fusion-for-Connected-Vehicles",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "imbalanced-learn>=0.11",
        "lightgbm>=4.0",
        "xgboost>=1.7",
        "shap>=0.42",
        "matplotlib>=3.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "predictive maintenance", "connected vehicles", "V2X",
        "edge computing", "LightGBM", "SHAP", "data fusion"
    ],
)
