
import ssl

# Tải dữ liệu MNIST
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import pickle
import mne
import logging
from Preprocessing import filter_data
from Preprocessing import STFT_transform
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải dữ liệu PhysioNet BCI2000
logging.info('Đang tải dữ liệu EEG...')



# Chuyển đổi dữ liệu EEG thành hình ảnh STFT
logging.info('Đang chuyển đổi dữ liệu EEG thành hình ảnh STFT...')


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

a = create_stft_matrix(vui)


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

models = [
    # LogisticRegression(),
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

stft_images = []

stft_images = np.concatenate((create_stft_matrix(vui), create_stft_matrix(buon)))


targets = pd.concat([pd.Series([0] * 586), pd.Series([1] * 586)]).values
# targets = np.array([0]* 586 + [1] )

def implement(train):
    if train == True:
        X = create_feature_matrix(stft_images)  # image_matrix_list là danh sách các ma trận hình ảnh
        y = targets  # labels là danh sách các nhãn tương ứng với từng hình ảnh


        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Huấn luyện SVM
        logging.info('Đang huấn luyện mô hình SVM...')

        logging.info('Đang trích xuất đặc trưng GLCM từ tập test...')


        logging.info('Đang dự đoán trên tập test...')
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization penalty
            'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in optimization
            'l1_ratio': np.linspace(0, 1, 10)  # Only used for 'elasticnet'
        }

        params = {'C': 100, 'l1_ratio': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}

        # Create the Logistic Regression model
        # logistic_regression = LogisticRegression(max_iter=100)

        # # Set up the grid search
        # grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

        # # Perform the grid search
        # grid_search.fit(X_train, Y_train)

        # # Retrieve the best parameters and score
        # best_params = grid_search.best_params_
        # best_score = grid_search.best_score_
        # print("Best parameters:", best_params)
        # print("Best cross-validation accuracy:", best_score)

        # Evaluate on the test set
        # model = grid_search.best_estimator_
        model = LogisticRegression(**params)

        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        test_score = model.score(X_test, Y_test)
        k = 10
        kf = KFold(n_splits=k)
        accuracy_list = []
        results = []
        for train_index, test_index in kf.split(X_train):
            # Split data into training and testing sets
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy)
        avg_accuracy = np.mean(accuracy_list)
        results.append((type(model).__name__, train_score, test_score, avg_accuracy))
        # plt.show()
        print(train_score)
        print(test_score)
        print(avg_accuracy)
        res = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Test Accuracy', 'K fold Accuracy'])
        print(res)



    else: ##load model
        pass

implement(train = True)
