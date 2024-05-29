import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import FeatureExtract, filter_data
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset


vui1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiVui.txt")
vui2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiVui2.txt")
vui3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/QuangVui3.txt")
vui4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonVui4.txt")
vui5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui2.txt")
vui6 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui.txt")

vui = np.concatenate((vui1, vui2, vui3, vui4, vui5, vui6))
print(vui.shape)

buon1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/BachBuon.txt")
buon2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiBuon.txt")
buon3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonBuon3.txt")
buon4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonBuon4.txt")
buon5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfBuon.txt")


buon = np.concatenate((buon1, buon2, buon3, buon4, buon5))

calm1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiCalm.txt")
calm2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/BachCalm.txt")
calm3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/QuangCalm3.txt")
calm4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiCalm2.txt")
calm5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfCalm.txt")


calm = np.concatenate((calm1, calm2, calm3, calm4, calm5))

vui = filter_data(vui)
buon = filter_data(buon)
calm = filter_data(calm)

df = pd.DataFrame.from_dict(FeatureExtract(vui, plot=0))
df1 = pd.DataFrame.from_dict(FeatureExtract(buon, plot=0))
df2 = pd.DataFrame.from_dict(FeatureExtract(calm, plot=0))

X = pd.concat([df, df1, df2]).values
print(X.shape)

# X = pd.concat([df, df1, df2])
print(X)
# 0 la nham mat, 1 la mo mat, 2 la tap trung
y = pd.concat([pd.Series([0] * len(df)), pd.Series([1] * len(df1)), pd.Series([2] * len(df2))]).values
# y = pd.concat([pd.Series([0] * len(df1)), pd.Series([1] * len(df2))]).values
print(y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)

# Define parameter grids
# param_grids = {
#     'LogisticRegression': {
#         'C': [0.1, 1, 10, 100],
#         'solver': ['liblinear', 'saga']
#     },
#     'DecisionTreeClassifier': {
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'sqrt', 'log2'],
#     'criterion': ['gini', 'entropy']    
#     },
#     'RandomForestClassifier': {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
#     },
#     'SVC': {
#         'C': [0.1, 1, 10, 100],
#         'kernel': ['linear', 'rbf'],
#         'gamma': ['scale', 'auto']
#     },
#     'KNeighborsClassifier': {
#         'n_neighbors': [3, 5, 7, 9],
#         'weights': ['uniform', 'distance'],
#         'metric': ['euclidean', 'manhattan']
#     },
#     'GaussianNB': {
#         # GaussianNB does not have hyperparameters that can be tuned using GridSearchCV.
#     },
#     'LinearDiscriminantAnalysis': {
#         'solver': ['svd', 'lsqr', 'eigen']
#     },
#     'AdaBoostClassifier': {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 1]
#     },
#     'GradientBoostingClassifier': {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7]
#     },
#     'XGBoostClassifier': {
#         'objective': ['multi:softmax'],  
#         'num_class': [3],  
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'n_estimators': [100, 200, 300],
#         'subsample': [0.7, 0.8, 0.9],
#         'colsample_bytree': [0.7, 0.8, 0.9],
#         # 'eval_metric': 'mlogloss'
#     }
# }

param_grids = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'max_iter': [500, 1000, 2000]
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'GaussianNB': {
        'var_smoothing': [1e-09, 1e-08, 1e-07]
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto']
    },
    'AdaBoostClassifier': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 1, 10]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 1, 10],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'XGBoostClassifier': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 1, 10],
        'max_depth': [None, 5, 10],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1],
        'colsample_bytree': [0.5, 0.8, 1],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
}

# Define models
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(), 
    'XGBoostClassifier': XGBClassifier()
}

# Perform Grid Search with K-fold Cross-Validation
best_estimators = {}
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for model_name in models:
    model = models[model_name]
    param_grid = param_grids.get(model_name, {})

    if not param_grid:  # Skip models with no hyperparameters to tune (e.g., GaussianNB)
        best_estimators[model_name] = model
        model.fit(X_train, Y_train)
        continue

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    best_estimators[model_name] = grid_search.best_estimator_

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validated accuracy for {model_name}: {grid_search.best_score_}")

# Evaluate the best models on the test set
for model_name in best_estimators:
    best_model = best_estimators[model_name]
    y_pred = best_model.predict(X_test)
    print(f"Test accuracy for {model_name}: {accuracy_score(Y_test, y_pred)}")
