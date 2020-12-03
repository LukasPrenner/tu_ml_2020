import numpy as np

def gradientDescent(X, Y, w_vector, alpha, n_iter=100):
    w_count = w_vector.shape[0]
    update_w = lambda of_w, w_value: w_value -  alpha * calculateDerivateResidualSumOfSquares(X, Y, w_vector, of_w)
    for iter in range(n_iter):
        w_vector = np.array(list(map(update_w, range(0,w_count), w_vector)))
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

def calculateWeightedAttributeSum(x_vector, w_vector): # equal to (w_0 + w_1 * x_1 + ... + w_n * x*n)
    w_0 = w_vector[0]
    w_for_zip = w_vector[1:]
    return w_0 + w_for_zip.dot(x_vector)


if __name__ == "__main__":
    raw_data = np.genfromtxt('testinput.txt')
    X_values = np.delete(raw_data, raw_data.shape[1]-1, 1)
    Y_values = raw_data[:,raw_data.shape[1]-1]
    dimensions = X_values.shape[1]
    weights = np.full(dimensions+1, 1.0)

    #print(raw_data)
    #print(weights)
    #print(calculateWeightedAttributeSum(X_values[0], weights))
    #print(calculateResidual(X_values[0], Y_values[0], weights))
    #print(calculateSquaredResidual(X_values[0], Y_values[0], weights))
    print("RSS:")
    print(calculateResidualSumOfSquares(X_values, Y_values, weights))
    print("Derivatives: ")
    print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 0))
    print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 1))
    print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 2))
    print(calculateDerivateResidualSumOfSquares(X_values, Y_values, weights, 3))
    print("Gradient Descent: ")
    print(gradientDescent(X_values, Y_values, weights, alpha=2.0, n_iter=110))
    