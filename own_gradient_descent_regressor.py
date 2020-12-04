import numpy as np

class OwnGradientDescentRegressor:
    def __init__(self, alpha=0.0001, max_iter=1000, n_iter_no_change=5, debug_output=False):
        self.w_vector = None
        self.alpha = alpha 
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.debug_output = debug_output

    def fit(self, X, Y, coef_init=None, intercept_init=1.0):
        self.initalizeWeights(X, coef_init, intercept_init)
        w_count = self.w_vector.shape[0]
        update_w = lambda of_w, w_value: w_value -  self.alpha * OwnGradientDescentRegressor.calculateDerivateResidualSumOfSquares(X, Y, self.w_vector, of_w)

        n_iter_no_change_count = 0
        for iter in range(self.max_iter):
            if (iter+1) % 100 == 0 and self.debug_output:
                print(f'Running iteration {iter+1}/{self.max_iter}') 

            w_prev = self.w_vector

            self.w_vector = np.array(list(map(update_w, range(0,w_count), self.w_vector)))

            n_iter_no_change_count = n_iter_no_change_count+1 if np.allclose(w_prev, self.w_vector, rtol=1e-30, atol=1e-35) else 0 #check if there is sequence of the same weights
            if((n_iter_no_change_count+1) == self.n_iter_no_change):
                if self.debug_output:
                    print(f'Iteration stopped at iteration={iter} since there was no change in weight in the last {self.n_iter_no_change} iterations!')
                return self.w_vector[1:], self.w_vector[0]
        if self.debug_output:
            print(f'Iteration did not converege with alpha={self.alpha}, max_iter={self.max_iter}, n_iter_no_change={self.n_iter_no_change}')
        return self.w_vector[1:], self.w_vector[0]
    
    def predict(self, X):
        return [OwnGradientDescentRegressor.calculateWeightedAttributeSum(x_vector, self.w_vector) for x_vector in X]
    
    def initalizeWeights(self, X, coef_init, intercept_init):
        if coef_init.any() == None:
            self.w_vector = np.full(X.shape[1], 1.0)
        else:
            self.w_vector = coef_init[0]
        self.w_vector = np.insert(self.w_vector, 0, intercept_init, axis=0) #prepend the intercept w_0

    @staticmethod
    def calculateDerivateResidualSumOfSquares(X, Y, w_vector, of_w): #works for w_1..w_n but not for w_0
        return sum([OwnGradientDescentRegressor.calculateDerivateSquaredResidualTerm(x_vector, y_value, w_vector, of_w) for x_vector, y_value in zip(X,Y)])

    @staticmethod
    def calculateDerivateSquaredResidualTerm(x_vector, y_value, w_vector, of_w):
        x_value_of_w = 1.0 if of_w == 0 else x_vector[of_w-1]
        outer_derivate = -2.0 * x_value_of_w
        return outer_derivate * OwnGradientDescentRegressor.calculateResidual(x_vector, y_value, w_vector)

    @staticmethod
    def calculateResidualSumOfSquares(X, Y, w_vector):
        return sum([OwnGradientDescentRegressor.calculateSquaredResidual(x_vector, y_value, w_vector) for x_vector, y_value in zip(X,Y)])

    @staticmethod
    def calculateSquaredResidual(x_vector, y_value, w_vector):
        return OwnGradientDescentRegressor.calculateResidual(x_vector, y_value, w_vector) ** 2.0

    @staticmethod
    def calculateResidual(x_vector, y_value, w_vector):
        return y_value - OwnGradientDescentRegressor.calculateWeightedAttributeSum(x_vector, w_vector) # equal to y - (w_0 + w_1 * x_1 + ... + w_n * x*n)

    @staticmethod
    def calculateWeightedAttributeSum(x_vector, w_vector): # equal to (w_0 + w_1 * x_1 + ... + w_n * x_n)
        w_0 = w_vector[0]
        w_for_zip = w_vector[1:]
        return w_0 + w_for_zip.dot(x_vector)