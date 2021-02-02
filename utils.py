import numpy as np

class LinearRegressionDataGenerator(object):
    """
    Data Generator for Linear Regression Model
    y = X * var + epsilon
    ================
    Member Variables
    ================
    var            : ground true parameter for linear regression model
    dim            : the dimension for the linear regression model
    std            : standard deviation for the noise
    """

    def __init__(self, var, std = 0.1):
        var = np.array(var)
        assert(var.ndim is 1)

        self.var = var
        self.dim = var.size
        self.std = std

    def generate(self, data_num):
        X = np.random.rand(data_num, self.dim)
        y = X.dot(self.var) + self.std * np.random.randn(data_num)
        return X, y

class LogisticRegressionDataGenerator(object):
    
    def __init__(self, var):
        var = np.array(var)
        assert(var.ndim is 1)

        self.var = var
        self.dim = var.size

    def generate(self, data_num):
        X = np.random.randn(data_num, self.dim)
        prob = 1 / (1 + np.exp(X.dot(self.var)))
        y = (np.random.rand(data_num) < prob).astype(np.int)
        y = y * 2 - 1
        return X, y

def eval_numerical_gradient(f, x):
    """ 
    a naive implementation of numerical gradient of f at x 
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """ 
    fx = f(x) # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 1e-8
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        new_x = np.copy(x)
        new_x[ix] += h # increment by h
        fxh = f(new_x) # evalute f(x + h)
        # compute the partial derivative
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimension
    return grad