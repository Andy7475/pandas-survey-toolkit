from setuptools import setup, find_packages

setup(
    name="pandas-survey-toolkit",
    version="0.1.0",
    author="Andy Laing",
    author_email="andylaing5@gmail.com",
    description="A pandas extension for survey analysis",
    url="https://github.com/Andy7475/pandas-survey-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/Andy7475/pandas-survey-toolkit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.0.0",
        "pandas-flavor>=0.2.0",
        "sentence_transformers",
        "numpy",
        "umap-learn",
        "scikit-learn",
    ],
)