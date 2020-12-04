import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

np.seterr('raise')

def gradientDescent(X, Y, w_vector, alpha=0.0001, max_iter=1000, n_iter_no_change=10):
    w_count = w_vector.shape[0]
    update_w = lambda of_w, w_value: w_value -  alpha * calculateDerivateResidualSumOfSquares(X, Y, w_vector, of_w)

    n_iter_no_change_count = 0
    for iter in range(max_iter):
        if (iter+1) % 100 == 0:
            print(f'Running iteration {iter+1}/{max_iter}') 

        w_prev = w_vector

        w_vector = np.array(list(map(update_w, range(0,w_count), w_vector)))

        n_iter_no_change_count = n_iter_no_change_count+1 if np.allclose(w_prev, w_vector, rtol=1e-30, atol=1e-35) else 0 #check if there is sequence of the same weights
        if((n_iter_no_change_count+1) == n_iter_no_change):
            print(f'Iteration stopped at iteration={iter} since there was no change in weight in the last {n_iter_no_change} iterations!')
            return w_vector
    print(f'Iteration did not converege with alpha={alpha}, max_iter={max_iter}, n_iter_no_change={n_iter_no_change}')
    return w_vector

def calculateDerivateResidualSumOfSquares(X, Y, w_vector, of_w): #works for w_1..w_n but not for w_0
    return sum([calculateDerivateSquaredResidualTerm(x_vector, y_value, w_vector, of_w) for x_vector, y_value in zip(X,Y)])

def calculateDerivateSquaredResidualTerm(x_vector, y_value, w_vector, of_w):
    x_value_of_w = 1.0 if of_w == 0 else x_vector[of_w-1]
    outer_derivate = -2.0 * x_value_of_w
    return outer_derivate * calculateResidual(x_vector, y_value, w_vector)

def calculateResidualSumOfSquares(X, Y, w_vector):
    return sum([calculateSquaredResidual(x_vector, y_value, w_vector) for x_vector, y_value in zip(X,Y)])

def calculateSquaredResidual(x_vector, y_value, w_vector):
    return calculateResidual(x_vector, y_value, w_vector) ** 2.0

def calculateResidual(x_vector, y_value, w_vector):
    return y_value - calculateWeightedAttributeSum(x_vector, w_vector) # equal to y - (w_0 + w_1 * x_1 + ... + w_n * x*n)

def calculateWeightedAttributeSum(x_vector, w_vector): # equal to (w_0 + w_1 * x_1 + ... + w_n * x_n)
    w_0 = w_vector[0]
    w_for_zip = w_vector[1:]
    return w_0 + w_for_zip.dot(x_vector)

def predicitWithWeights(X, w_vector):
    return [calculateWeightedAttributeSum(x_vector, w_vector) for x_vector in X]

if __name__ == "__main__":
    #raw_data = np.genfromtxt('datasets/metro.csv', delimiter=',')
    #raw_data = pd.read_csv('datasets/metro.csv')
    raw_data = pd.read_excel('datasets/real_estate.xlsx').to_numpy()
    #raw_data = StandardScaler().fit_transform(raw_data)
    X_values = np.delete(raw_data, raw_data.shape[1]-1, 1)
    Y_values = raw_data[:,raw_data.shape[1]-1]
    dimensions = X_values.shape[1]
    weights = np.full(dimensions+1, 1.0)

    print(StandardScaler().fit_transform(raw_data))
    print(raw_data.shape)

    print(weights)
    #print(calculateWeightedAttributeSum(X_values[0], weights))
    #print(calculateResidual(X_values[0], Y_values[0], weights))
    #print(calculateSquaredResidual(X_values[0], Y_values[0], weights))
    #print("RSS:")
    #print(calculateResidualSumOfSquares(X_values, Y_values, weights))
    #print("Derivatives: ")
    #print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 0))
    #print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 1))
    #print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 2))
    #print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 3))
    #print("Gradient Descent: ")
    #weights = gradientDescent(X_values, Y_values, weights, alpha=0.0001, max_iter=1000)
    #print([3.44790020e-15,1.06713711e-01,-2.25812564e-01,-4.16251964e-01,2.45344636e-01,2.05646664e-01,-1.40191268e-02])
    #print(X_values[0])

    gdc = SGDRegressor()
    gdc.fit(X_values, Y_values)
    print(gdc.predict([X_values[0]]))   
    print(predicitWithWeights([X_values[0]], weights))
    print(Y_values[0])

    #print(predicitWithWeights([X_values[0]], weights))
    #print(Y_values[0])

    