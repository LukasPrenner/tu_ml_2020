from sklearn.preprocessing import StandardScaler
from own_knn_regressor import OwnKNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time


if __name__ == "__main__":
    raw_data = pd.read_excel('datasets/real_estate.xlsx')
    #raw_data = pd.read_excel('datasets/real_estate.xlsx').to_numpy()
    #raw_data= pd.read_csv('datasets/metro.csv')

    #raw_data = pd.DataFrame(StandardScaler().fit_transform(raw_data), columns=raw_data.columns)
    #raw_data = StandardScaler().fit_transform(raw_data) # TO NOT RUN INTO OVERFLOW ERRORS WITH OUR IMPLEMENTATION -> ALWAYS SCALE(!)

    X_values = raw_data.iloc[:, :-1]
    Y_values = raw_data.iloc[:, -1]
    #X_values = np.delete(raw_data, raw_data.shape[1] - 1, 1)
    #Y_values = raw_data[:, raw_data.shape[1] - 1]

    X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values, test_size=0.3, random_state=33)
    X_train = X_train.reset_index().drop(['index'], axis=1)
    X_test = X_test.reset_index().drop(['index'], axis=1)
    Y_train = Y_train.reset_index().drop(['index'], axis=1)
    Y_test = Y_test.reset_index().drop(['index'], axis=1)
    X_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(StandardScaler().fit_transform(X_test), columns=X_test.columns)
    tic = time.clock()
    own_knn = OwnKNeighborsRegressor(n_neighbors=5)
    y_hat_own_knn = own_knn.findknearestNeighbors(X_train, Y_train, X_test)
    toc = time.clock()
    #for i in y_hat_own_knn:
    #    if(np.isnan(i)):
    #        print(i)
    print(y_hat_own_knn)
    print("Time own implementation: " + str(toc - tic))

    tic = time.clock()
    knn = KNeighborsRegressor()
    knn.fit(X_train,Y_train)
    y_hat_knn = knn.predict(X_test)
    toc = time.clock()
    print("Time own implementation: " + str(toc - tic))
    print(y_hat_knn)

    #res = np.array(sorted(np.abs(y_hat_own_knn-y_hat_knn)))
    #print(res[res > 1e-10])