from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name="pandas-survey-toolkit",
    # version="1.0.0", being sorted by github
    author="Andy Laing",
    author_email="andylaing5@gmail.com",
    description="A pandas extension for survey analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0,<3.0",  # Need 2.1+ for torch.compiler support
        "sentence-transformers>=3.0,<4.0",
        "umap-learn>=0.5,<1.0",
        "scikit-learn>=1.3,<2.0",  # Updated to allow 1.7.x
        "pandas>=2.2.0,<3.0",
        "numpy>=1.26.0,<2.0",
        "pandas-flavor>=0.6.0,<1.0",
        "spacy>=3.0.0,<4.0",
        "gensim>=4.3.3,<5.0",  # Fixed version for scipy 1.13+ compatibility
        "scipy>=1.10.0,<1.15.0",  # Compatible with gensim
        "altair>=4.0.0,<5.0",
        "matplotlib>=3.0.0,<4.0",
        "pyvis>=0.3.2,<1.0",
        "transformers>=4.20.0,<5.0",  # Add explicit transformers constraint
    ],
    license="MIT",
    license_files=["LICENSE"],
)
