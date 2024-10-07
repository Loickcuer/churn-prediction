# Churn Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# 🚀 Churn Prediction for Retail Banking

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 📊 Project Overview

This advanced data science project tackles a critical challenge in retail banking: customer churn prediction. Leveraging a comprehensive dataset of customer information and departure history, I've developed a sophisticated scoring model to identify at-risk customers and provide actionable insights for retention strategies.

### 🎯 Key Objectives

- Develop a high-accuracy predictive model for customer churn
- Generate individual risk scores for each customer
- Provide explainable AI insights on churn factors
- Empower bank advisors with data-driven, personalized retention strategies

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning modeling
- **XGBoost**: Advanced gradient boosting
- **MLFlow**: Model interpretability
- **Streamlit**: Interactive web application

## 🚀 Quick Start

1. **Clone the repository:**
```
git clone https://github.com/Loickcuer/churn-prediction.git
```

2. **Set up the environment:**
   
```
pip install -r requirements.txt
```
3. **Run the main script:**
   
Scripts must be executed from the churn_prediction folder, not the root folder.

Launching the dataset cleaning, the modeling and the streamlit app :
```
python main.py
```

## 🖥️ Streamlit App Demo

Experience our interactive Churn Prediction Dashboard:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loickcuer-churn-prediction-docsstreamlit-app-susqjw.streamlit.app/)

## 📂 Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── churn_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes churn_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

