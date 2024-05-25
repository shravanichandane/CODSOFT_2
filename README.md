# TASK - 2

# Fraud Detection Model

## Project Overview

This project focuses on building a machine learning model to detect fraudulent transactions. The main steps include data preparation, model building, hyperparameter tuning, model evaluation, and feature importance analysis.

## Dataset

The dataset used in this project consists of transaction data with various features such as transaction time, user demographics, and transaction details. You can get the Datasets [HERE](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

## Tools and Libraries

The following tools and libraries were used in this project:

- **Python**: Programming language used for implementing the model.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **scikit-learn**: For machine learning model building and evaluation.
  - `LogisticRegression`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `GridSearchCV`
  - `train_test_split`
  - `StandardScaler`
  - `LabelEncoder`
  - `Pipeline`
  - `classification_report`
  - `accuracy_score`
- **Matplotlib**: For plotting feature importances.
- **zipfile**: For handling zip files.

## Data Preparation
The data cleaning process was performed using the Data Wrangler extension in VS Code, which provided an intuitive interface for manipulating and transforming the dataset.
The `prepare_data` function handles data preprocessing:
- Convert date columns to datetime objects.
- Calculate the age of users.
- Extract the hour from the transaction time.
- Encode categorical variables.
- Drop unnecessary columns.

## Model Building

Three different models are built and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

Each model is trained and evaluated using a pipeline that includes data scaling and the chosen model.

## Model Evaluation

The models are evaluated on accuracy, precision, recall, and F1-score using the test dataset.

### Logistic Regression
```text
Accuracy: 0.9954869277458571
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.00      0.00      0.00      2145

    accuracy                           1.00    555719
   macro avg       0.50      0.50      0.50    555719
weighted avg       0.99      1.00      0.99    555719
```

### Decision Tree
```text
Accuracy: 0.9979989886975252
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.72      0.79      0.75      2145

    accuracy                           1.00    555719
   macro avg       0.86      0.89      0.88    555719
weighted avg       1.00      1.00      1.00    555719
```

### Random Forest
```text
Accuracy: 0.998808750465613
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.95      0.73      0.83      2145

    accuracy                           1.00    555719
   macro avg       0.97      0.86      0.91    555719
weighted avg       1.00      1.00      1.00    555719
```

### Validation Set Performance
```text
Accuracy: 0.9986195461468756
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    257815
           1       0.97      0.79      0.87      1520

    accuracy                           1.00    259335
   macro avg       0.98      0.89      0.93    259335
weighted avg       1.00      1.00      1.00    259335
```

## Code 
You can get the code [HERE](https://github.com/shravanichandane/CODSOFT_2/blob/main/CODSOFT_2.ipynb) in the same respository.

## Feature Importance

The feature importances of the best model (Random Forest) are analyzed and visualized to understand which features contribute most to the model's predictions.

## Results

Based on the evaluation metrics, the Random Forest model performed the best, with the following highlights:
- High accuracy and precision in detecting fraud.
- The model is able to generalize well on the validation set.
- Key features contributing to the model's performance include:
