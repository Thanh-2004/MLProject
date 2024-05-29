import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import FeatureExtract, filter_data, STFT_transform
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

from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte



models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

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
vui5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui.txt")

vui = np.concatenate((vui1, vui2, vui3, vui4, vui5))
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

def SplitArray(arr):

# Mảng ban đầu
    # arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Kích thước của các mảng con và số phần tử trùng lặp
    subarray_size = 15*512
    overlap = 14*512

    # Khởi tạo danh sách để lưu các mảng con
    subarrays = []

    # Sử dụng vòng lặp để tạo các mảng con
    for i in range(0, len(arr) - subarray_size + 1, subarray_size - overlap):
        subarray = arr[i:i + subarray_size]
        subarrays.append(subarray)

    # Chuyển danh sách các mảng con thành mảng NumPy
    subarrays = np.array(subarrays)

    print("Original array:", arr)
    print("Subarrays with overlap:")
    # print(subarrays)
    return subarrays


# splitted_vui = SplitArray(vui)

def create_stft_matrix(data):
    data_segments = SplitArray(data)
    stft_matrices = []
    for segment in data_segments:
        stft_matrices.append(STFT_transform(segment))
    stft_matrices = np.array((stft_matrices))
    return stft_matrices



def normalize_image(image):
    """
    Chuẩn hóa ảnh về khoảng [0, 1]
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if min_val == max_val:
        return np.zeros_like(image)
    else:
        return (image - min_val) / (max_val - min_val)

glcm_feature = []
def compute_glcm(image_matrix):
    # Chuyển đổi ma trận hình ảnh STFT sang dạng ảnh xám (ubyte)
    # gray_image = image_matrix
    image_matrix = normalize_image(image_matrix)

    gray_image = img_as_ubyte(image_matrix)

    # gray_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
 

    # Tính GLCM cho ảnh xám
    distances = [1]  # Các khoảng cách giữa các pixel
    angles = [0, 40*np.pi/180, 95*np.pi/180, 135*np.pi/180]  # Các góc quay
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    return glcm

# Sử dụng:

def extract_features(glcm):
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    return features

# 3. Tạo ma trận đặc trưng cho các hình ảnh
def create_feature_matrix(images):
    features = []
    for image in images:
        glcm = compute_glcm(image)
        image_features = extract_features(glcm)
        features.append(image_features)
    return np.array(features)

stft_images = []

stft_images = np.concatenate((create_stft_matrix(vui), create_stft_matrix(buon), create_stft_matrix(calm)))


targets = pd.concat([pd.Series([0] * 586), pd.Series([1] * 586), pd.Series([2] * 586) ]).values

X = create_feature_matrix(stft_images)
y = targets

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)

# Define parameter grids
param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'GaussianNB': {
        # GaussianNB does not have hyperparameters that can be tuned using GridSearchCV.
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen']
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoostClassifier': {
        'objective': ['multi:softmax'],  
        'num_class': [3],  
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        # 'eval_metric': 'mlogloss'
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
