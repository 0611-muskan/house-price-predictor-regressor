# 🏠 California House Price Prediction

## 📌 Project Overview

This project focuses on predicting **house prices in California** using machine learning techniques. The dataset is sourced from **scikit-learn’s California Housing dataset**, and the project demonstrates a complete workflow including data preprocessing, outlier handling, model training, and evaluation.

This is a **regression-based machine learning project** suitable for academic submissions, GitHub portfolios, and ML fundamentals practice.

---

## 🎯 Objective

To build a machine learning model that accurately predicts **median house prices** based on various socioeconomic and geographical features such as income, population, and location.

---

## 📊 Dataset Description

* **Dataset Source:** `sklearn.datasets.fetch_california_housing`
* **Type:** Built-in dataset (no external download required)
* **Target Variable:** `price` (Median house value)

### 🔑 Features

* `MedInc` – Median income in block group
* `HouseAge` – Median house age
* `AveRooms` – Average number of rooms
* `AveBedrms` – Average number of bedrooms
* `Population` – Population of block group
* `AveOccup` – Average occupancy
* `Latitude` – Latitude
* `Longitude` – Longitude

---

## 🧹 Data Cleaning & Preprocessing

The following preprocessing steps are performed:

* Conversion of dataset into a Pandas DataFrame
* Column renaming and formatting
* Duplicate record detection
* Missing value analysis (dataset contains no major null values)
* Outlier detection using:

  * Statistical summary (`describe()`)
  * Boxplot visualization
* Outlier removal using **Z-score method**
* Cleaning column names (removal of extra spaces)

These steps ensure data quality and improve model performance.

---

## 🧠 Machine Learning Model

* **Algorithm Used:** Random Forest Regressor
* **Why Random Forest?**

  * Handles non-linear relationships well
  * Robust to outliers
  * Provides better accuracy compared to basic linear models

---

## ⚙️ Model Training Process

* Feature–target separation
* Train-test split (80% training, 20% testing)
* Model fitting on training data
* Prediction on test data

---

## 📈 Model Evaluation

The model is evaluated using:

* **R² Score** – Measures goodness of fit
* **Mean Squared Error (MSE)** – Measures prediction error

These metrics help assess how well the model generalizes to unseen data.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run the Project

1. Clone the repository
2. Install required libraries
3. Run the Python script

```bash
python california_house_price_prediction.py
```

---

## 📌 Key Learnings

* Practical data preprocessing techniques
* Outlier handling using Z-score
* Regression modeling with Random Forest
* Model evaluation using standard metrics

---


