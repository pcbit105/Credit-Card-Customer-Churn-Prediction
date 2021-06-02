# Credit-Card-Customer-Churn-Prediction
This is a machine Learning project for predicting customer churn
The dataset is from Kaggle (https://www.kaggle.com/c/1056lab-credit-card-customer-churn-prediction/overview). The dataset includes features such as customer demographics (e.g. age, gender, education level, marital status, income level), credit card status and spending (e.g. card type, months inactive, credit limit, total revolving balance, transaction amount).
Algorithms used are Logistic Regression, K-Nearest Neighbors, Kernel SVM, Naive Bayes, Random Forest, Artificial Neural Network, XGBoost, and CatBoost.
Using the Optuna library for hyperparameter optimization, I have found the CatBoost algorithm as the best performer (AUC - 99.46%), with XGBoost as a close 2nd (AUC - 99.36%).
