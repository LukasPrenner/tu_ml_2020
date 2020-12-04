from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from own_gradient_descent_regressor import OwnGradientDescentRegressor
import pandas as pd
import numpy as np

if __name__ == "__main__":
    #raw_data = np.genfromtxt('datasets/metro.csv', delimiter=',')
    raw_data = pd.read_excel('datasets/real_estate.xlsx').to_numpy() # ALWAYS USE NUMPY FOR OUR IMPLEMENTATION NOT PANDAS!!
    raw_data = StandardScaler().fit_transform(raw_data) # TO NOT RUN INTO OVERFLOW ERRORS WITH OUR IMPLEMENTATION -> ALWAYS SCALE(!)

    X_values = np.delete(raw_data, raw_data.shape[1]-1, 1)
    Y_values = raw_data[:,raw_data.shape[1]-1]

    weights_sk = np.full((1,X_values.shape[1]), 1.0) #do not reuse the weights since sk-learn does inplace work with the coef_init matrix!
    intercept_sk = 1
    weights_own = np.full((1,X_values.shape[1]), 1.0)
    intercept_own = 1

    sk_gdc = SGDRegressor()
    sk_gdc.fit(X_values, Y_values, coef_init=weights_sk, intercept_init=intercept_sk) #coef_init is the same as our weights for comparison reasons (sklear does not pass w_0!)
    print("Weights and intercept found by sk:", weights_sk, intercept_sk)

    own_gdc = OwnGradientDescentRegressor(debug_output=False)
    weights_own, intercept_own = own_gdc.fit(X_values, Y_values, coef_init=weights_own, intercept_init=intercept_own)
    print("Weights and intercept found by own:",weights_own, intercept_own)

    print("Prediction with skLearn:", sk_gdc.predict([X_values[0]])[0])   
    print("Prediction with own-imp:", own_gdc.predict([X_values[0]])[0])
    print("Actual y-value:         ",Y_values[0])
