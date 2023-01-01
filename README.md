![Python 3.9.0](https://img.shields.io/badge/Python-3.9.0-blue.svg)
![Conda](https://img.shields.io/conda/vn/conda-forge/python?color=green)
![Umap](https://img.shields.io/badge/Packages-umap-green.svg)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

<h1> Classification of five types of cancer based on RNA-seq data using machine learning </h1>

## Goals  
Create ML models to predict the classification of different types of cancer based on a matrix of gene expression  
Test different ML algorithms  
Test how different parameters can affect the model

## Setting-up conda env
```{}
conda create --name projectCancerClassifier python=3.9
conda activate projectCancerClassifier
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=projectCancerClassifier
conda install -c conda-forge scikit-learn
conda install -c conda-forge umap-learn
conda install -c conda-forge seaborn
conda upgrade -c conda-forge scikit-learn
```
## Running the script
Either by running
```{}
python3 main.py
```
Or by opening the jupyter notebook file. It's better to follow the jupyter notebook file rather than running the script on one go.

## Report
The report of this study is availabe in PDF format.
