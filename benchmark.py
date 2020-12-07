
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM, SVR
from sklearn.linear_model import Ridge, SGDRegressor, Lasso
from own_gradient_descent_regressor import OwnGradientDescentRegressor

RSEED = 123

def split_data(X,y):
    kfold = KFold(n_splits=10, random_state=RSEED, shuffle=True)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    # reset index
    X_train = X_train.reset_index().drop(['index'], axis=1)
    X_test = X_test.reset_index().drop(['index'], axis=1)
    
    scaled_features_train = X_train.copy()
    scaled_features_test = X_test.copy()

    # only select numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    columns_to_scale = X_train.select_dtypes(include=numerics).columns
    
    features_to_scale_train = scaled_features_train[columns_to_scale]
    features_to_scale_test = scaled_features_test[columns_to_scale]
    
    scaler = StandardScaler()
    scaler.fit(features_to_scale_train)
    scaled_features = pd.DataFrame(scaler.transform(features_to_scale_train))
    scaled_features_train[columns_to_scale] = scaled_features
    scaled_features = pd.DataFrame(scaler.transform(features_to_scale_test))
    scaled_features_test[columns_to_scale] = scaled_features
    
    return scaled_features_train, scaled_features_test

def process_missing_values(X_train, y_train):
    # drop missing values
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    
    return X_train, y_train

def process_outliers(X_train, y_train):
    # only select numeric columns
    numerics = ['uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X_train = X_train.select_dtypes(include=numerics)

    # identify outliers in the training dataset
    outlier_predictor = OneClassSVM(nu=0.02)
    y_hat = outlier_predictor.fit_predict(X_train)
    unique, counts = np.unique(y_hat, return_counts=True)

    # select all rows that are not outliers
    outlier_mask = y_hat != -1
    return X_train[outlier_mask], y_train[outlier_mask]

def prepare_real_estate_data(data):
    data = data.iloc[:,1:]
    x_columns = data.iloc[:,0:6].columns.str[3:]
    y_column = data.iloc[:,6:7].columns.str[2:]
    columns = x_columns.append(y_column)
    data.columns = columns
    X = data.iloc[:,0:6]
    y = data.iloc[:,6:7]
    return data, X, y

def preprocess_data(X_train, X_test, y_train, y_test):
    X_train, y_train = process_outliers(X_train, y_train)
    X_train, y_train = process_missing_values(X_train, y_train)
    X_train, X_test = scale_data(X_train, X_test)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    #raw_data = np.genfromtxt('datasets/metro.csv', delimiter=',')
    #raw_data = pd.read_excel('datasets/real_estate.xlsx').to_numpy() # ALWAYS USE NUMPY FOR OUR IMPLEMENTATION NOT PANDAS!!
    #raw_data = StandardScaler().fit_transform(raw_data) # TO NOT RUN INTO OVERFLOW ERRORS WITH OUR IMPLEMENTATION -> ALWAYS SCALE(!)
    data = pd.read_excel("data/dataset_RealEstateValuation.xlsx")

    data_prepared, X, y = prepare_real_estate_data(data)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    #print(X_train)
    #X_values = np.delete(raw_data, raw_data.shape[1]-1, 1)
    #Y_values = raw_data[:,raw_data.shape[1]-1]

    weights_sk = np.full((1,X_train.shape[1]), 1.0) #do not reuse the weights since sk-learn does inplace work with the coef_init matrix!
    intercept_sk = 1
    weights_own = np.full((1,X_train.shape[1]), 1.0)
    intercept_own = 1


    sk_gdc = SGDRegressor()
    sk_gdc.fit(X_train, y_train, coef_init=weights_sk, intercept_init=intercept_sk) #coef_init is the same as our weights for comparison reasons (sklear does not pass w_0!)
    print("Weights and intercept found by sk:", weights_sk, intercept_sk)

    own_gdc = OwnGradientDescentRegressor(debug_output=True)
    print(weights_own, weights_own.shape)
    weights_own, intercept_own = own_gdc.fit(X_train, y_train, coef_init=weights_own, intercept_init=intercept_own)
    print("Weights and intercept found by own:",weights_own, intercept_own)

    print("Prediction with sk-learn:", sk_gdc.predict(X_test))
    print("Prediction with own-imp:", own_gdc.predict(X_test))

