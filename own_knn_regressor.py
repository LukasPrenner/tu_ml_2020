import numpy as np
import pandas as pd

class OwnKNeighborsRegressor:
    def __init__(self, n_neighbors=5, p=2):
        self.p = p
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        print("fit")

    def predict(self, X):
        print("predict")


    def findknearestNeighbors(self, X, Y, X_test):
        y_hat = []
        XY = pd.concat([X, Y],axis=1)
        #print(X)
        #print(Y)
        #print(XY)


        for point_test in X_test.iterrows():
            distances = []

            for point_train in XY.iterrows():
                #print(point_train)
                distance = self.calculateDistance(point_test[1], point_train[1][:-1], self.p)
                distances.append([distance, point_train[1][-1]])
                #print(distances)


            distances_df = pd.DataFrame(data=distances)

            nearest = distances_df.sort_values(by= 0, axis=0)[0:self.n_neighbors]


            y_hat.append(nearest[1].mean())


        return y_hat

    def calculateDistance(self, x, y, p=1):
        dim = len(x)

        distance = 0

        for i in range(dim):
            distance += abs(x[i] - y[i])**p

        distance = distance**(1/p)

        return distance