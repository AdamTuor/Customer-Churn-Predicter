# Telecom Customer Churn Prediction and Web Application
## Participants
 - Adam Tuor
 - Aditya Parmar
 - Gurpreet Singh Dhameja
 - Jasper Bowen Pang
 - Marco Teh

## Table of Contents

- [Introduction](#introduction)
- [Data Exploration](#data-exploration)
- [Machine Learning Model](#machine-learning-model)
- [Web Application Demo](#web-application-demo)
- [Installation](#installation)
- [Usage](#usage)
- [SHAP](#shap)

## Introduction

This project aims to predict customer churn in a telecom company and develop a web application to showcase the predictions in real-time. By identifying customers who are likely to churn, the company can take proactive measures to retain them and reduce customer attrition.

## Data Exploration

To gain insights into the dataset, we performed extensive data exploration. We analyzed various features, visualized data distributions, and explored correlations between different variables. This process helped us understand the characteristics of churned and non-churned customers and identify important patterns or trends.

## Machine Learning Model

### Model Overview

We developed a machine learning model using a Random Forest Classifier algorithm. Random Forest is well-suited for classification tasks and offers good accuracy and robustness against overfitting. It creates an ensemble of decision trees and leverages their collective predictions to make accurate classifications.

### Training Process

The training process involved several steps:

1.	Data Preprocessing: We handled missing values and addressed data types in the dataset, ensuring that the data was in a suitable format for analysis.

2.	Train-Test Split: We split the dataset into training and testing sets, with an 80:20 ratio, to train the model on a portion of the data and evaluate its performance on unseen data.

3.	Encoding Categorical Variables: We used OneHotEncoder to encode categorical variables, converting them into binary representations that could be understood by the machine learning model.

4.	Scaling Numerical Features: We employed StandardScaler to scale numerical features, standardizing them to have a mean of 0 and a standard deviation of 1. This helped ensure that all features contributed equally to the model's learning process.

5.	Model Training: We trained a Random Forest Classifier using the training set, optimizing its parameters to learn patterns and relationships within the data.

6.	Model Evaluation: We evaluated the performance of the trained model using various metrics such as accuracy, precision, recall, and F1-score. This allowed us to assess the model's predictive capability and measure its effectiveness in predicting customer churn.

### Integration into the Web Application

The trained machine learning model was seamlessly integrated into the web application. The application allows users to input customer information, such as gender, contract type, monthly charges, etc. The input data is preprocessed, transformed, and fed into the model for churn prediction. The prediction results are then displayed to the user in real-time, providing insights into whether a customer is likely to churn or not.

## Web Application Demo

Our web application provides a user-friendly interface for interacting with the churn prediction model. It offers the following features:

- Input form for capturing customer details
- Real-time prediction of customer churn
- Visualizations of prediction results
- [Demo Website](https://customer-churn-predictor.onrender.com/)

## Installation

To run the web application locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Usage

1. Access the web application through the provided URL or by running it locally.
2. Fill in the required customer details in the input form.
3. Click the "Predict" button to obtain churn prediction results.
4. View the prediction outcome and accompanying visualizations on the web application interface.
5. Explore the different sections and functionalities of the application to gain insights into customer churn.

## SHAP

The SHAP force plot consists of a set of bars that represent different features, and each bar represents the contribution of that feature to the final prediction. Here's what the user can observe:

- **Y-axis:** The vertical axis represents the features. Each feature is labeled with its name and encoded/scaled value.

- **Color:** The color of each bar indicates the direction and magnitude of the feature's impact on the prediction. Positive contributions are displayed in red, indicating they push the prediction towards a positive outcome (e.g., "The customer will churn" in our case). Negative contributions are displayed in blue, indicating they push the prediction towards a negative outcome (e.g., "The customer will not churn" in our case).

- **Length:** The length of each bar represents the magnitude or strength of the feature's impact on the prediction. Longer bars indicate features with a larger influence on the prediction.

- **Base value:** The plot also includes a reference line, which represents the expected or average prediction. Features that extend to the right of the line have a positive impact, while features that extend to the left have a negative impact.

By examining the SHAP force plot, the user can identify which features contribute the most to the prediction and understand their individual effects. They can see which features push the prediction towards a positive outcome ("Yes") and which features push it towards a negative outcome ("No"). The length and color of the bars help the user gauge the magnitude and direction of each feature's influence on the prediction.


