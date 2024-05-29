
import ssl

# Tải dữ liệu MNIST
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from sklearn.model_selection import train_test_split, KFold
import pickle
import logging
from Preprocessing import STFT_transform
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

from DataRetrieve import vui, buon, calm

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải dữ liệu PhysioNet BCI2000
logging.info('Đang tải dữ liệu EEG...')



# Chuyển đổi dữ liệu EEG thành hình ảnh STFT
logging.info('Đang chuyển đổi dữ liệu EEG thành hình ảnh STFT...')


def SplitArray(arr):

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

    return subarrays


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
    image_matrix = normalize_image(image_matrix)

    gray_image = img_as_ubyte(image_matrix)

 

    # Tính GLCM cho ảnh xám
    distances = [1]  # Các khoảng cách giữa các pixel
    angles = [0, 40*np.pi/180, 95*np.pi/180, 135*np.pi/180]  # Các góc quay
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    return glcm


def extract_features(glcm):
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    return features

def create_feature_matrix(images):
    features = []
    for image in images:
        glcm = compute_glcm(image)
        image_features = extract_features(glcm)
        features.append(image_features)
    return np.array(features)



stft_images = []

stft_images = np.concatenate((create_stft_matrix(vui), create_stft_matrix(buon), create_stft_matrix(calm)))


targets = pd.concat([pd.Series([0] * 586), pd.Series([1] * 586), pd.Series([2]*586)]).values


X = create_feature_matrix(stft_images)
y = targets
   

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
    results.append((type(model).__name__, train_score, score, avg_accuracy))
    print("train_score: ", train_score)
    print("test_score: ", score)
    print("avg_accuracy: ", avg_accuracy)
    print(model.predict(X_test))    
    
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

params = {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
model = KNeighborsClassifier(**params)
models.append(model)

params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_class': 3, 'objective': 'multi:softmax', 'subsample': 0.8}
model = XGBClassifier(**params)
models.append(model)

params = {'criterion': 'entropy', 'max_depth': 30, 'min_samples_split': 5}
model = DecisionTreeClassifier(**params)
models.append(model)

params = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
model = RandomForestClassifier(**params)
models.append(model)
label_mapping = {'NEGATIVE': 1, 'NEUTRAL': 2, 'POSITIVE': 0}

for model in models:
    TrainingModel(model)
    filename = f'SourceCode/FinalModel/trained_model/{type(model).__name__}_GLCM.h5'
    pickle.dump(model, open(filename, 'wb'))
res = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Test Accuracy', 'K fold Accuracy'])
print(res)
