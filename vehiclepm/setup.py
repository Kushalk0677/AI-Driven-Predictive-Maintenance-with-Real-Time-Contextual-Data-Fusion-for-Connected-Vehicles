from setuptools import setup, find_packages
import os

# README is optional — don't fail if not found
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if not os.path.exists(readme_path):
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")

try:
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Predictive maintenance for connected vehicles."

setup(
    name="vehiclepm",
    version="0.2.0",
    author="Kushal Khemani",
    author_email="kushal.khemani@gmail.com",
    description=(
        "Predictive maintenance for connected vehicles with V2X contextual "
        "data fusion, SHAP interpretability, noise sensitivity analysis, "
        "and live OBD-II inference."
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
        "requests>=2.28",
    ],
    extras_require={
        "obd": ["obd>=0.7", "pyserial>=3.5"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "isort"],
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
        "edge computing", "LightGBM", "SHAP", "data fusion", "OBD2"
    ],
)
