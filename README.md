# Fraud-Detection-AI
Fraud Detection on transactions.The agenda for this code is to train a model that can predict whether a transaction is fraudulent or not based on various features in the dataset such as age, gender, card type, transaction amount, etc.



# Import necessary libraries:
sklearn.ensemble for RandomForestClassifier
sklearn.linear_model for LogisticRegression
sklearn for svm
matplotlib.pyplot as plt
sklearn.tree for DecisionTreeClassifier
sklearn.metrics for various evaluation metrics
pandas as pd
Load the data from a CSV file using pd.read_csv and assign it to the variable data.

Split the data into features (X) and target variable (y).

Split the data into training and testing sets using train_test_split from sklearn.model_selection.

Print information about the data, check for missing values, and identify categorical and continuous variables.

Encode categorical data using one-hot encoding using pd.get_dummies.

Perform standard scaling on the continuous variables using StandardScaler from sklearn.preprocessing.

Split the encoded and scaled data into training and testing sets again.

Train and evaluate various models:

Logistic Regression: Fit the model, make predictions, and calculate evaluation metrics.
Support Vector Machine (SVM): Fit the model, make predictions, and calculate evaluation metrics.

K-Nearest Neighbors (KNN): Fit the model, make predictions, and calculate evaluation metrics. Also, iterate over different values of k and calculate metrics for each.
Decision Tree: Fit the model, make predictions, and calculate evaluation metrics.
Random Forest: Fit the model, make predictions, and calculate evaluation metrics.
Gradient Boosting Classifier: Fit the model, make predictions, and calculate evaluation metrics.
Save the trained Random Forest model using joblib.dump.

Load the saved model using joblib.load and make predictions on new data.

Create a graphical user interface (GUI) using Tkinter to input values and make predictions using the trained model.


# Code explanation

The provided code performs fraud detection using various machine learning
models.
It starts by loading a dataset and preprocessing it by encoding categorical
variables and standardizing continuous variables.
The dataset is then split into training and testing sets. The code implements
several classification models including Logistic Regression, Support Vector
Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, Random Forest,
and Gradient Boosting.
Model performance metrics such as accuracy, recall, precision, F1-score, and
balanced accuracy are calculated and displayed.
Additionally, there is a GUI application that allows users to input their own
data for fraud detection.


P.S please have a look at the presentation for a better understanding of the code and the implementation of graphs.
the data set is of 1026 entries and from kaggle
