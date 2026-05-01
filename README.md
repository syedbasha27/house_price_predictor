# 🏠 Home Price Prediction using XGBoost

## 📌 Project Overview

This project predicts house prices using the **XGBoost regression algorithm**.
The main objective is to estimate property prices based on housing features and **engineered socio-economic proxy variables** such as income, school quality, hospital access, and crime rate.

---

## 🎯 Problem Statement

Predict home prices using factors like:

* Income
* Schools
* Hospitals
* Crime rate

⚠️ Since the dataset did not include these directly, **proxy features were created using data analysis techniques**.

---

## 📊 Dataset Description

The dataset contains housing-related features such as:

* Bedrooms, Bathrooms
* Square footage (living & lot area)
* Floors, Condition, Grade
* Location (zipcode, latitude, longitude)
* Year built & renovated
* Price (Target Variable)

---

## 🧠 Feature Engineering (Key Highlight)

To address missing real-world factors, the following proxy features were created using **pandas**:

### 💰 Income Proxy

* Calculated as **average house price per zipcode**
* Represents economic strength of an area

### 🏫 School Quality Proxy

* Based on **average house grade per zipcode**
* Higher grade → better school zones

### 🏥 Hospital Access Proxy

* Derived from **distance to city center**
* Closer distance → better access to facilities

### 🚔 Crime Rate Proxy

* Inverse of income proxy
* Lower income areas → higher assumed crime rate

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost

---

## 🚀 Implementation Steps

1. Load dataset using pandas
2. Clean data (remove unnecessary columns, handle missing values)
3. Perform feature engineering to create proxy variables
4. Select relevant features
5. Split dataset into training and testing sets
6. Train XGBoost Regressor model
7. Evaluate model using performance metrics (MAE, RMSE, R²)
8. Predict house prices

---

## 📈 Model Performance

The model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Results indicate that incorporating proxy features improves prediction performance.

---

## 🔍 Key Insights

* `sqft_living` and `grade` are strong predictors
* Location-based features significantly impact price
* Proxy features effectively simulate real-world socio-economic conditions

---

## 💡 Future Improvements

* Integrate real datasets for income, crime, and school ratings
* Perform hyperparameter tuning for better accuracy
* Deploy using Streamlit or a web application

---

## 🧠 Conclusion

This project demonstrates how **feature engineering and proxy variables** can solve real-world problems when direct data is unavailable.
XGBoost effectively captures complex relationships and provides accurate house price predictions.

---

## 👨‍💻 Author

**Your Name Here**

---

## 📌 Note

This project was developed as part of an internship task focusing on machine learning and data analysis.
