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
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=3.0",
        "umap-learn>=0.5",
        "scikit-learn>=1.5",
        "pandas>=2.2.0",
        "numpy==1.24.3",
        "pandas-flavor>=0.6.0",
        "spacy>=3.7",
        "gensim>=4.3.3",
        "altair>=5.4.0",
    ],
    license="MIT",
    license_files=["LICENSE"],
)
