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
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


models = [
    LogisticRegression(penalty = 'l2', C = 0.0001),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    # MLPClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    # XGBClassifier(),
    # LGBMClassifier()
]


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


df = pd.DataFrame.from_dict(FeatureExtract(vui, plot=0))
df1 = pd.DataFrame.from_dict(FeatureExtract(buon, plot=0))
df2 = pd.DataFrame.from_dict(FeatureExtract(calm, plot=0))

X = pd.concat([df, df1]).values
print(X.shape)

# X = pd.concat([df, df1, df2])
print(X)
# 0 la nham mat, 1 la mo mat, 2 la tap trung
y = pd.concat([pd.Series([0] * len(df)), pd.Series([1] * len(df1))]).values
# y = pd.concat([pd.Series([0] * len(df1)), pd.Series([1] * len(df2))]).values
print(y.shape)
# EDA data



def TrainingModel(model):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)
    model.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    k = 10
    kf = KFold(n_splits=k)
    loo = LeaveOneOut()
    accuracy_list = []
    for train_index, test_index in kf.split(X_train):
        # print(train_index, test_index)
        # Split data into training and testing sets
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
    score = model.score(X_test, Y_test)
    accuracy_list.append(score)
    avg_accuracy = np.mean(accuracy_list)
    results.append((type(model).__name__, train_score, test_score, avg_accuracy))
    print("train_score: ", train_score)
    print("test_score: ", test_score)
    print("avg_accuracy: ", avg_accuracy)

results = []
models = []

params = {'criterion': 'gini', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 18}
model = RandomForestClassifier(**params)
models.append(model)

params = {'max_depth': 12, 'min_samples_leaf': 1, 'min_samples_split': 2}
model = DecisionTreeClassifier(**params)
models.append(model)


for model in models:
    TrainingModel(model)
    filename = 'test.h5'
    pickle.dump(model, open(filename, 'wb'))
    break
res = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Test Accuracy', 'K fold Accuracy'])
print(res)



# print("Test score: ", score)
# filename = 'test.h5'
# pickle.dump(model, open(filename, 'wb'))

