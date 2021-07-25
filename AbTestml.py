import mlflow
import dvc.api as dvc
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os


path_v5 = pd.read_csv("data/v5.csv")
path_v6 = pd.read_csv("data/v6.csv")
version_5 = 'v5'
version_6 = "v6"

data_v5_url = dvc.get_url(
    path=path_v5,
    repo="/home/iboy/PycharmProjects/10Academy",
    rev="v5"
)


mlflow.set_experiment("10Academy")
data_v5 = pd.read_csv(data_v5_url, sep=",")
X = data_v5[:, 0:6]
y = data_v5[:, 7]


# XGBoost Model on version 5
def xgboost_algo():

    mlflow.log_param('data_url', data_v5_url)
    mlflow.log_param('version', version_5)
    mlflow.log_param('input_rows', data_v5.shape[0])
    mlflow.log_param('input_cols', data_v5.shape[1])

    # train _ test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    # validate
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    model = XGBClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    pred = [round(value) for value in y_pred]

    #evaluate
    acc = accuracy_score(y_test, pred)
    print("Accuracy score for XGBoost:", acc)



# Decision Tree Algorithm
def decisiontree_algo():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # validate
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy score for decision Tree:", accuracy_score(y_test, y_pred))

# Decision Tree Algorithm
def logisticsRegression_algo():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # validate
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    lr = LogisticRegression()

    # Train Decision Tree Classifer
    clf = lr.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy score for Logistics Regression: ", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    xgboost_algo()
    decisiontree_algo()
    logisticsRegression_algo()








