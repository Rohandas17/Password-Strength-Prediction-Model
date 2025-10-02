# Password Strength Prediction Model 🔐

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)

## Introduction
This project presents a regression model designed to predict the strength of passwords. Using a dataset of common passwords, the model leverages an XGBoost Regressor to determine a password's strength based on engineered features. The workflow includes crucial preprocessing steps like extracting features (e.g., length, character types) from raw password strings, handling categorical data, and scaling the target variable to build an efficient and accurate predictive model.

## Features  
- **XGBoost Regression:** Implements a powerful gradient boosting algorithm (XGBoost) to accurately model the relationship between password characteristics and their strength.
- **Custom Feature Engineering:** Extracts key predictive features from each password, including its length and the counts of uppercase letters, lowercase letters, digits, and special characters.
- **Categorical Data Handling:** Uses one-hot encoding to convert the password category feature into a numerical format, allowing the model to interpret it effectively.
- **Feature Scaling:** Normalizes the target variable (password strength) using MinMaxScaler to ensure a consistent scale for model training and evaluation. 

## Installation & Usage
**1. Clone the repository:**
```bash
git clone https://github.com/Rohandas17/Password-Strength-Prediction-Model
```
**2. Installation:**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

**3. Usage:**
```bash
python main.py
```

## Contributing
This project is open source so others can easily get involved. If you'd like to contribute, please fork the repository, create a feature branch, and open a pull request. All kinds of contributions bug fixes, features, or suggestions — are welcome!
