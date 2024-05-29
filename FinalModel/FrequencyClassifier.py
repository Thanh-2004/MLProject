import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocessing import FeatureExtract, filter_data
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from DataRetrieve import vui, buon, calm

# vui = filter_data(vui)
# buon = filter_data(buon)
# calm = filter_data(calm)

df = pd.DataFrame.from_dict(FeatureExtract(vui, plot=0))
df1 = pd.DataFrame.from_dict(FeatureExtract(buon, plot=0))
df2 = pd.DataFrame.from_dict(FeatureExtract(calm, plot=0))


X = pd.concat([df, df1, df2]).values
print(X.shape)


y = pd.concat([pd.Series([0] * len(df)), pd.Series([1] * len(df1)), pd.Series([2] * len(df2))]).values
print(y.shape)




def TrainingModel(model):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)
    model.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    k = 10
    kf = KFold(n_splits=k)
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
    print("test_score: ", score)
    print("avg_accuracy: ", avg_accuracy)
    y_pred = np.array(model.predict(X_test))

    cm = confusion_matrix(Y_test, y_pred)
    clr = classification_report(Y_test, y_pred, target_names=label_mapping.keys())

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
    plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
    plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)

results = []
models = []

# params = {'criterion': 'gini', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 18}
params =  {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
model = RandomForestClassifier(**params)
models.append(model)

# params = {'criterion': 'gini', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 18}
params = {'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5}
model = DecisionTreeClassifier(**params)
models.append(model)

label_mapping = {'NEGATIVE': 1, 'NEUTRAL': 2, 'POSITIVE': 0}

for model in models:
    TrainingModel(model)
    filename = f'SourceCode/FinalModel/trained_model/{type(model).__name__}_Frequency.h5'
    pickle.dump(model, open(filename, 'wb'))
res = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Test Accuracy', 'K fold Accuracy'])
print(res)

