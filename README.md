# autovehiclepricing
Auto vehicle pricing evaluation

# 🚗 Luxury Vehicle Pricing Prediction System

> **An End-to-End MLOps Pipeline for Automated, Data-Driven Vehicle Pricing**

This repository contains the code and infrastructure for an MLOps pipeline designed to solve pricing inefficiencies for an automobile dealership in Los Vegas. By leveraging machine learning, this system automates the evaluation of luxury and non-luxury vehicles, ensuring accurate, consistent, and scalable pricing decisions.

## 📌 The Problem

The dealership currently relies on manual, disconnected systems for pricing. This leads to:
*   **Inconsistency:** Human error in evaluating complex variables like mileage, engine power, and segment.
*   **Latency:** Slow updates to pricing strategies in response to market shifts.
*   **Scalability Issues:** Difficulty handling growing demand without proportional increases in operational overhead.

## 🎯 The Solution

We have implemented a **Continuous Integration/Continuous Delivery (CI/CD) enabled MLOps pipeline** that automates the entire lifecycle of a pricing model.

### Key Features
*   **Automated Data Pipeline:** Ingests raw vehicle data, handles cleaning, preprocessing, and feature engineering.
*   **Model Training & Evaluation:** Builds and evaluates regression models to predict vehicle price based on technical specifications.
*   **Model Registry:** Automatically registers trained models with performance metrics for tracking and versioning.
*   **Scalable Deployment:** Designed to integrate seamlessly with production environments, ensuring real-time or batch pricing updates.

## 📊 Dataset Overview

The model is trained on historical vehicle data. The target variable is **Price**, predicted based on the following features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| **Segment** | Vehicle category (Luxury vs. Non-Luxury) | Categorical |
| **Kilometers_Driven** | Total distance driven | Numerical |
| **Mileage** | Fuel efficiency (km/l) | Numerical |
| **Engine** | Engine capacity (cc) | Numerical |
| **Power** | Engine power (BHP) | Numerical |
| **Seats** | Number of seats | Numerical |
| **Price** | **Target Variable**: Vehicle cost (in Lakhs) | Numerical |

## 🏗️ Project Architecture

The pipeline follows a standard MLOps workflow:

1.  **Data Ingestion:** Fetch data from source.
2.  **Preprocessing:** Handle missing values, encode categorical features (`Segment`), and scale numerical features.
3.  **Model Training:** Train a regression model (e.g., Linear Regression, Random Forest, or XGBoost) to minimize pricing error.
4.  **Evaluation:** Validate model performance using metrics such as MAE (Mean Absolute Error) and RMSE (Root Mean Square Error).
5.  **Registration:** Store the model artifact and metadata in the Model Registry.
6.  **CI/CD:** GitHub Actions (or similar) triggers the pipeline on code changes, ensuring that only validated models are promoted to production.



## 📈 Model Performance

The model is evaluated on its ability to predict vehicle prices accurately. Key metrics tracked include:
*   **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual price.
*   **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily.
*   **R² Score:** Proportion of variance in pricing explained by the model.

*(Note: Update this section with your actual model metrics after training.)*

## 🛠️ Technologies Used

*   **Language:** Python
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **MLOps Tools:** MLflow (for model registry/tracking), Docker
*   **CI/CD:** GitHub Actions
*   **Version Control:** Git


## 👥 Contributors

*   (https://github.com/santsada) - *MLOps Engineer*

