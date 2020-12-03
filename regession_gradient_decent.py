import numpy as np



def calculateRSS(X, Y, w_vector):
    return sum([calculateSquaredResidual(x_vector, y_value, w_vector) for x_vector, y_value in zip(X,Y)])

def calculateSquaredResidual(x_vector, y_value, w_vector):
    return calculateResidual(x_vector, y_value, w_vector) ** 2

def calculateResidual(x_vector, y_value, w_vector):
    return y_value - calculateWeightedAttributeSum(x_vector, w_vector)

def calculateWeightedAttributeSum(x_vector, w_vector): # equal to w_0 + w_1 * x_1 + ... + w_n * x*n
    w_0 = w_vector[0]
    w_for_zip = w_vector[1:]
    return w_0 + w_for_zip.dot(x_vector)


if __name__ == "__main__":
    raw_data = np.genfromtxt('testinput.txt')
    X_values = np.delete(raw_data, raw_data.shape[1]-1, 1)
    Y_values = raw_data[:,raw_data.shape[1]-1]
    dimensions = X_values.shape[1]
    weights = np.full(dimensions+1, 1.0)

    print(weights)
    print(calculateWeightedAttributeSum(X_values[0], weights))
    print(calculateResidual(X_values[0], Y_values[0], weights))
    print(calculateSquaredResidual(X_values[0], Y_values[0], weights))
    print(calculateRSS(X_va0-lues, Y_values, weights))
    