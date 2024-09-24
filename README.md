# Churn Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to develop a predictive system for a retail bank facing a customer attrition problem. Using a dataset indexing customer information and departure history over three months, the goal is to create a scoring model capable of identifying customers likely to leave the bank. The system must not only generate an attrition risk score but also provide explanations about the factors influencing this risk, thus allowing bank advisors to adapt their commercial approach. The project encompasses in-depth statistical analyses, the construction of a robust predictive model, and the development of result interpretation tools, aiming to optimize customer retention through a data-driven and personalized approach.

# Installation

Use a dedicated environnement to install the librairies.

Clone repo :
```
git clone https://github.com/Loickcuer/churn-prediction.git
```

Install the requirements for the app from the root folder of the project:
```
pip install -r requirements.txt
```

Scripts must be executed from the churn_prediction folder, not the root folder.

Launching the dataset cleaning, the modeling and the streamlit app :
```
python main.py
```

## Streamlit App

You can have a look at the streamlit app serving as an interface for the bank advisors to use. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loickcuer-churn-prediction-docsstreamlit-app-susqjw.streamlit.app/)

## Project Organization

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

