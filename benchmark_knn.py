from sklearn.preprocessing import StandardScaler
from own_knn_regressor import OwnKNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


if __name__ == "__main__":
    raw_data = pd.read_excel('datasets/real_estate.xlsx')

    raw_data = pd.DataFrame(StandardScaler().fit_transform(raw_data), columns=raw_data.columns) # TO NOT RUN INTO OVERFLOW ERRORS WITH OUR IMPLEMENTATION -> ALWAYS SCALE(!)

    X_values = raw_data.iloc[:, :-1]
    Y_values = raw_data.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values, test_size=0.3, random_state=33)
    own_knn = OwnKNeighborsRegressor()
    y_hat_own_knn = own_knn.findknearestNeighbors(X_train, Y_train, X_test)
    #print(y_hat_own_knn)


    knn = KNeighborsRegressor()
    knn.fit(X_train,Y_train)
    y_hat_knn = knn.predict(X_test)
    #print(y_hat_knn)

    res = np.array(sorted(np.abs(y_hat_own_knn-y_hat_knn)))
    print(res[res > 1e-10])