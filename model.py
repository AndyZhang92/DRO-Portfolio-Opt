import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from config import Configuration
from utils import eval_numerical_gradient

class GradientDescent(object):
    """
    Abstract Model for Gradient Descent
    ================
    Member Variables
    ================
    learning_rate (float) : Learning rate for gradient descent algorithm.
    max_iter      (int)   : Max num of iteration of gradient descent method.
    total_iter    (int)   : Total num of iteration in training.
    tol           (float) : If the change in the loss function <= tol in the
                            last 100 iterations, early terminate the model.
    var        np.array[dim]       : Initialization of the variables of the
                                     optimization problem.
    dim        int                 : Dimension of the variable
    loss_path  np.array[max_iter]  : Path of the loss function during gradient
                                     descent.
    var_path   np.ndarray[max_iter,dim]: Path of the variable during training
    """

    def __init__(self, **kwargs):
        # Initialization using the default configuration
        self.learning_rate = Configuration.learning_rate
        self.max_iter = Configuration.max_iter
        self.total_iter = Configuration.max_iter
        self.tol = Configuration.tol
        self.model_name = Configuration.model_name

        # Initialization using user input value from **kwargs
        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]
            else:
                raise AttributeError("Unknown Initialization of {}"
                                     .format(key))
        # Check Initialization of variable var
        if "var" not in self.__dict__:
            raise NotImplementedError("var is not initialized")
        if type(self.var) is not np.ndarray or self.var.ndim > 1:
            raise TypeError("var should be 1-dim np.ndarray")
        
        self.dim = self.var.size
        self.loss_path = np.zeros(self.max_iter, dtype=float)
        self.var_path = np.zeros((self.max_iter, self.dim), dtype=float)

    def gradient(self, method = "analytic"):
        """
        Compute the Gradient for the Model
        """
        if method is "analytic":
            return self._analytical_gradient()
        elif method is "numeric":
            return eval_numerical_gradient(self.loss, self.var)
        else:
            raise ValueError("Unknown Method to Compute Gradient: {}"
                             .format(method))
    def _analytical_gradient(self):
        """
        Compute the Analytical Gradient for the Model
        """
        raise NotImplementedError("""Analytical gradient is not implemented. 
        Please use numerical gradient instead.""")
    
    def loss(self):
        """
        Loss Functiton for the Model
        """
        raise NotImplementedError("""Loss is not implemented.""")

    def project(self):
        """
        Compute the Projection for Projected Gradient Descent Method
        """
        pass

    def optimize(self, verbose=False, grad_method='analytic'):
        """
        Run the Gradient Descent Algorithm for the model
        """
        for iter_ in range(self.max_iter):
            gradient = self.gradient(method=grad_method)
            self.var -= self.learning_rate * gradient
            self.project()
            loss = self.loss()
            self.loss_path[iter_] = loss
            self.var_path[iter_, :] = self.var
            if 'min_lambda_path' in dir(self):
                self.min_lambda_path[iter_] = self.min_lambda
                
            if verbose and iter_ % 1 == 0:
                print("Step:{}\tLoss = {}".format(iter_, loss))
                print("Var:\n{}".format(self.var))
                if 'min_lambda' in dir(self):
                    print("Min Lambda:\n{}".format(self.min_lambda))
                print("Gradient:\n{}".format(gradient))
                
            if iter_ > 100 and (self.loss_path[iter_-100] -
                                self.loss_path[iter_] <= self.tol):
                self.total_iter = iter_
                if verbose:
                    print("Early Break at step {}".format(iter_))
                break

    def plot_loss_path(self):
        """
        Plot the loss path of the model after training
        """
        x = range(1, self.total_iter + 1)
        plt.plot(x, self.loss_path[:self.total_iter])
        plt.title(self.model_name)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()

class LinearRegressionGradientDescent(GradientDescent):
    """
    Gradient Descent for Linear Regression Model (No intercept)
    ================
    Member Variables
    ================
    learning_rate (float) : Learning rate for gradient descent algorithm.
    max_iter      (int)   : Max num of iteration of gradient descent method.
    tol           (float) : If the change in the loss function <= tol in the
                            last 100 iterations, early terminate the model.
    X       np.ndarray[num_train,dim]:training data for linear regression model.
    y       np.array[num_train]    : The response vec for training.
    var        np.array[dim]       : Initialization of the variables of the
                                     optimization problem.
    dim        int                 : Dimension of the variable
    loss_path  np.array[max_iter]  : Path of the loss function during gradient
                                     descent.
    var_path   np.ndarray[max_iter,dim]: Path of the variable during training
    """

    def __init__(self, X, y, **kwargs):
        self.X = np.array(X)
        self.y = np.array(y)
        self.var = np.zeros(X.shape[1])
        super().__init__(**kwargs)
        # Check the dim of the data
        assert(self.X.shape[0] == self.y.size)
        assert(self.X.shape[1] == self.var.size)
        
    def _analytical_gradient(self):
        """
        Compute the Analytical Gradient for the Model
        """
        return np.mean(2 * (self.X.dot(self.var) - self.y) * self.X.T, axis=1)

    def loss(self, var=None):
        """
        Loss Function for Linear Regression Model
        Compute the Mean Squared Error for Linear Regression
        If var is None, compute loss at self.var
        Otherwise compute loss at var
        """
        if var is None:
            var = self.var
        return np.square(self.X.dot(var) - self.y).mean()

    def project(self):
        """
        No Projection for LR
        """
        pass

class DistRobustGradientDescent(GradientDescent):
    """
    Abstract Model for Gradient Descent
    ================
    Member Variables
    ================
    learning_rate (float) : Learning rate for gradient descent algorithm.
    max_iter      (int)   : Max num of iteration of gradient descent method.
    tol           (float) : If the change in the loss function <= tol in the
                            last 100 iterations, early terminate the model.
    var        np.ndarray[dim]     : Initialization of the variables of the
                                     optimization problem.
                                     The last dim represents the lagrange multiplier
                                     lambda
    X          np.ndarray          : Training data
                                     X[num_train, dim - 1] if fit_intercept = False
                                     X[num_train, dim - 2] if fit_intercept = True
    dim        int                 : Dimension of the variable (including lambda)
    A          np.ndarray: The local Mahalanobis matrice for DRO
    ------------------------------------------------------------------------------
    If A == None    ========>> A = 1, no Mahalanobis distance, using squared 
                                 Euclidean distance.
    If A.ndim == 2  ========>> A[dim-1, dim-1] Mahalanobis distance if fit_intercept = False
                               A[dim-2, dim-2] if fit_intercept = True
    If A.ndim == 3  ========>> A[dim-1, dim-1, num_train] if fit_intercept = False
                               A[dim-2, dim-2, num_train] if fit_intercept = True
                               where A[:,:,i] is the local Mahalanobis matrix
                               that corresponds to the ith data point.
    ------------------------------------------------------------------------------
    delta      float               : The size of distributional uncertainty set
    gamma      float               : The parameter used in the inner_max problem
    
    loss_path  np.ndarray[max_iter]: Path of the loss function during gradient
                                     descent.
    var_path   np.ndarray[max_iter,dim]: Path of the variable during training
    """
    
    def __init__(self, X, y = None, **kwargs):
        # Subclass of DistRobustGradientDescent should initialize var
        # in their __init__.
        self.X = np.array(X)
        self.num_train = X.shape[0]
        self.y = np.zeros(self.num_train) if y is None else np.array(y)
        self.fit_intercept = Configuration.fit_intercept
        self.delta = Configuration.delta
        self.gamma = Configuration.gamma
        self.A = np.array(Configuration.A)
        self.vardim = X.shape[1] + 1 + self.fit_intercept
        self.var = np.zeros(self.vardim)
        self.var[-1] = 0.5
        super().__init__(**kwargs)
        
        self.lambda_ = self.var[-1]
        if self.fit_intercept:
            self.beta = self.var[:-2]
            self.intercept = self.var[-2]
        else:
            self.beta = self.var[:-1]
            self.intercept = 0
        
        self._check_dimension()
        
        # Computation of invA
        if self.A.ndim == 0:
            self.invA = 1 / self.A
        elif self.A.ndim == 2:
            self.invA = np.linalg.inv(self.A)
        elif self.A.ndim == 3:
            self.invA = np.zeros_like(self.A)
            for i in range(self.A.shape[2]):
                self.invA[:, :, i] = np.linalg.inv(self.A[:, :, i])
    
    def l_func(self, x):
        """
        The l function that maps from R to R.
        The nonrobust loss is l(beta.T * x)
        """
        return np.square(x)
    
    def non_robust_loss(self, var=None):
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        return self.l_func(self.X.dot(beta) + intercept - self.y)
        
    def loss(self, var=None):
        """
        The robustified loss function for distributionally robust optimization
        """
        # Unpack Parameters
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        # Ancillary Variable: anc1 = sqrt(delta) * beta^T * inv(A) * beta
        anc1 = np.sqrt(self.delta) * beta.dot(beta.dot(self.invA))
        
        def max_obj(gamma):
            result = (self.l_func(self.X.dot(beta) + intercept + gamma*anc1 - self.y)
                     - lambda_*np.square(gamma)*anc1 + lambda_*np.sqrt(self.delta))
            return result
        
        opt_gamma = np.zeros(self.num_train)
        opt_val = np.zeros(self.num_train)
        for data_id in range(self.num_train):
            obj = lambda gamma: -self.l_func(self.X[data_id,:].dot(beta) + intercept \
                                            + gamma*anc1 - self.y[data_id]) \
            + lambda_*np.square(gamma)*anc1 - lambda_*np.sqrt(self.delta)
            res = minimize(obj, 0)
            opt_gamma[data_id] = res.x
            opt_val[data_id] = -res.fun
        return opt_val.mean()
        
    def _analytical_gradient(self):
        pass
    
    def project(self):
        pass
    
    def plot_lambda_path(self):
        """
        Plot the path of lambda
        """
        x = range(1, self.total_iter + 1)
        plt.plot(x, self.var_path[:self.total_iter, -1])
        plt.title(self.model_name)
        plt.xlabel('Steps')
        plt.ylabel('Lambda')
        plt.show()
        
    def _check_dimension(self):
        # Check Dimension:
        if self.A.ndim not in (0,2,3):
            raise ValueError("The dimension of A should be 0, 2 or 3.")
        if self.X.ndim != 2:
            raise ValueError("The input data X should be a matrix")
        beta_dim = self.dim - 2 if self.fit_intercept else self.dim - 1
        
        if self.A.ndim in (2,3):
            assert(beta_dim == self.A.shape[0] == self.A.shape[1] == self.X.shape[1])
        if self.A.ndim is 3:
            assert(self.train_num == self.A.shape[2])

class DistRobustLinearRegression(GradientDescent):
    """
    Abstract Model for Gradient Descent
    ================
    Member Variables
    ================
    learning_rate (float) : Learning rate for gradient descent algorithm.
    max_iter      (int)   : Max num of iteration of gradient descent method.
    tol           (float) : If the change in the loss function <= tol in the
                            last 100 iterations, early terminate the model.
    var        np.ndarray[dim]     : Initialization of the variables of the
                                     optimization problem.
                                     The last dim represents the lagrange multiplier
                                     lambda
    X          np.ndarray          : Training data
                                     X[num_train, dim - 1] if fit_intercept = False
                                     X[num_train, dim - 2] if fit_intercept = True
    dim        int                 : Dimension of the variable (including lambda)
    A          np.ndarray: The local Mahalanobis matrice for DRO
    ------------------------------------------------------------------------------
    If A == None    ========>> A = 1, no Mahalanobis distance, using squared 
                                 Euclidean distance.
    If A.ndim == 2  ========>> A[dim-1, dim-1] Mahalanobis distance if fit_intercept = False
                               A[dim-2, dim-2] if fit_intercept = True
    If A.ndim == 3  ========>> A[dim-1, dim-1, num_train] if fit_intercept = False
                               A[dim-2, dim-2, num_train] if fit_intercept = True
                               where A[:,:,i] is the local Mahalanobis matrix
                               that corresponds to the ith data point.
    ------------------------------------------------------------------------------
    delta      float               : The size of distributional uncertainty set
    gamma      float               : The parameter used in the inner_max problem
    
    loss_path  np.ndarray[max_iter]: Path of the loss function during gradient
                                     descent.
    var_path   np.ndarray[max_iter,dim]: Path of the variable during training
    """
    
    def __init__(self, X, y = None, **kwargs):
        # Subclass of DistRobustGradientDescent should initialize var
        # in their __init__.
        self.X = np.array(X)
        self.num_train = X.shape[0]
        self.y = np.zeros(self.num_train) if y is None else np.array(y)
        self.fit_intercept = Configuration.fit_intercept
        self.delta = Configuration.delta
        self.gamma = Configuration.gamma
        self.A = np.array(Configuration.A)
        self.vardim = X.shape[1] + 1 + self.fit_intercept
        self.var = np.zeros(self.vardim)
        self.var[-1] = 5
        super().__init__(**kwargs)
        
        self.lambda_ = self.var[-1:]
        if self.fit_intercept:
            self.beta = self.var[:-2]
            self.intercept = self.var[-2:-1]
        else:
            self.beta = self.var[:-1]
            self.intercept = 0
        
        self._check_dimension()
        
        # Computation of invA
        if self.A.ndim == 0:
            arrays = [np.eye(X.shape[1]) / self.A  for _ in range(self.num_train)]
            self.invA = np.stack(arrays, axis = 2)
        elif self.A.ndim == 2:
            arrays = [np.linalg.inv(self.A)  for _ in range(self.num_train)]
            self.invA = np.stack(arrays, axis = 2)
        elif self.A.ndim == 3:
            self.invA = np.zeros_like(self.A)
            for i in range(self.A.shape[2]):
                self.invA[:, :, i] = np.linalg.inv(self.A[:, :, i])
    
    def l_func(self, x):
        """
        The l function that maps from R to R.
        The nonrobust loss is l(beta.T * x)
        """
        return np.square(x)
    
    def non_robust_loss(self, var=None):
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        return self.l_func(self.X.dot(beta) + intercept)
        
    def loss(self, var=None):
        """
        The robustified loss function for distributionally robust optimization
        """
        # Unpack Parameters
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        # Ancillary Variable: anc1 = sqrt(delta) * beta^T * inv(A) * beta
        transport_X = self.transport_X(var = var)
        opt_gamma = self.opt_gamma(var = var)
        loss = np.mean(self.l_func(transport_X.dot(beta) + intercept - self.y)) \
        - lambda_ * np.sqrt(self.delta) * (np.mean(np.square(opt_gamma) * beta.dot(beta.dot(self.invA))) - 1)
        return loss
    
    def opt_gamma(self, var=None):
        # Unpack Parameters
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        anc1 = np.sqrt(self.delta) * beta.dot(beta.dot(self.invA))
        return (self.X.dot(beta) + intercept - self.y) / (lambda_ - anc1)
    
    def transport_X(self, var=None):
        # Unpack Parameters
        if var is None:
            var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        else:
            if self.fit_intercept:
                beta, intercept, lambda_ = var[:-2], var[-2:-1], var[-1:]
            else:
                beta, intercept, lambda_ = var[:-1], 0, var[-1:]
        opt_gamma = self.opt_gamma(var)
        res = self.X + np.transpose(np.sqrt(self.delta) * opt_gamma * beta.dot(self.invA))
        return res
        
    def _analytical_gradient(self):
        var, beta, intercept, lambda_ = self.var, self.beta, self.intercept, self.lambda_
        transport_X = self.transport_X()
        opt_gamma = self.opt_gamma()
        grad_beta = 2 * np.transpose((transport_X.dot(beta) + intercept - self.y) * transport_X.T)
        grad_lambda = -np.sqrt(self.delta) * (np.square(opt_gamma)*beta.dot(beta.dot(self.invA)) - 1)
        if self.fit_intercept:
            grad_intercept = 2 * (transport_X.dot(beta) + intercept - self.y)
            return np.append(np.mean(grad_beta, axis = 0), [np.mean(grad_intercept), np.mean(grad_lambda)])
        else:
            return np.append(np.mean(grad_beta, axis = 0), np.mean(grad_lambda))
        
    def project(self):
        pass
    
    def plot_lambda_path(self):
        """
        Plot the path of lambda
        """
        x = range(1, self.total_iter + 1)
        plt.plot(x, self.var_path[:self.total_iter, -1])
        plt.title(self.model_name)
        plt.xlabel('Steps')
        plt.ylabel('Lambda')
        plt.show()
        
    def _check_dimension(self):
        # Check Dimension:
        if self.A.ndim not in (0,2,3):
            raise ValueError("The dimension of A should be 0, 2 or 3.")
        if self.X.ndim != 2:
            raise ValueError("The input data X should be a matrix")
        beta_dim = self.dim - 2 if self.fit_intercept else self.dim - 1
        
        if self.A.ndim in (2,3):
            assert(beta_dim == self.A.shape[0] == self.A.shape[1] == self.X.shape[1])
        if self.A.ndim is 3:
            assert(self.train_num == self.A.shape[2])

class DistRobustMarkowitz(GradientDescent):
    """
    Distributionally Robust Model for Markowitz Mean-Variance Model
    ================
    Member Variables
    ================
    learning_rate (float) : Learning rate for gradient descent algorithm.
    max_iter      (int)   : Max num of iteration of gradient descent method.
    tol           (float) : If the change in the loss function <= tol in the
                            last 100 iterations, early terminate the model.
    var        np.ndarray[dim]     : Initialization of the variables of the
                                     optimization problem.
                                     The last dim represents the lagrange multiplier
                                     lambda
    X          np.ndarray[num_train, dim - 2]: Training data (historical return of stocks)
    dim        int                 : Dimension of the variable (including c and lambda)
    A          np.ndarray: The local Mahalanobis matrice for DRO
    ------------------------------------------------------------------------------
    If A == None    ========>> A = 1, no Mahalanobis distance, using squared 
                                 Euclidean distance.
    If A.ndim == 2  ========>> A[dim-2, dim-2] 
    If A.ndim == 3  ========>> A[dim-2, dim-2, num_train]
    ------------------------------------------------------------------------------
    delta      float               : The size of distributional uncertainty set
    gamma      float               : The parameter used in the inner_max problem
    
    loss_path  np.ndarray[max_iter]: Path of the loss function during gradient
                                     descent.
    var_path   np.ndarray[max_iter,dim]: Path of the variable during training
    """
    
    def __init__(self, X, **kwargs):
        # Subclass of DistRobustGradientDescent should initialize var
        # in their __init__.
        self.X = np.array(X)
        self.regularization = Configuration.regularization
        self.num_train = X.shape[0]
        self.delta = Configuration.delta
        self.gamma = Configuration.gamma
        self.A = np.array(Configuration.A)
        self.vardim = X.shape[1] + 2
        self.var = np.ones(self.vardim) / self.vardim
        self.var[-2] = 0
        self.var[-1] = self.delta * 10
        
        super().__init__(**kwargs)
        self.min_lambda_path = np.zeros(self.max_iter, dtype=float)
        
        self.lambda_ = self.var[-1:]
        self.c = self.var[-2:-1]
        self.beta = self.var[:-2]
        
        self._check_dimension()
        
        # Computation of invA
        if self.A.ndim == 0:
            arrays = [np.eye(X.shape[1]) / self.A  for _ in range(self.num_train)]
            self.invA = np.stack(arrays, axis = 2)
        elif self.A.ndim == 2:
            arrays = [np.linalg.inv(self.A)  for _ in range(self.num_train)]
            self.invA = np.stack(arrays, axis = 2)
        elif self.A.ndim == 3:
            self.invA = np.zeros_like(self.A)
            for i in range(self.A.shape[2]):
                self.invA[:, :, i] = np.linalg.inv(self.A[:, :, i])
        
    def l_func(self, x, c):
        """
        The l function that maps from R to R.
        The nonrobust loss is l(beta.T * x)
        """
        return np.square(x-c) - self.regularization * x
    
    def non_robust_loss(self, var=None):
        if var is None:
            var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        else:
            beta, c, lambda_ = var[:-2], var[-2:-1], var[-1:]
        return self.l_func(self.X.dot(beta), c)
        
    def loss(self, var=None):
        """
        The robustified loss function for distributionally robust optimization
        """
        # Unpack Parameters
        if var is None:
            var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        else:
            beta, c, lambda_ = var[:-2], var[-2:-1], var[-1:]
            
        # Ancillary Variable: anc1 = sqrt(delta) * beta^T * inv(A) * beta
        transport_X = self.transport_X(var = var)
        opt_gamma = self.opt_gamma(var = var)
        loss = np.mean(self.l_func(transport_X.dot(beta), c)) \
        - lambda_ * np.sqrt(self.delta) * (np.mean(np.square(opt_gamma) * beta.dot(beta.dot(self.invA))) - 1)
        return loss
    
    def compute_min_lambda(self, var=None):
        if var is None:
            var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        else:
            beta, c, lambda_ = var[:-2], var[-2:-1], var[-1:]
        anc1 = np.sqrt(self.delta) * beta.dot(beta.dot(self.invA))
        return np.max(anc1)
    
    def opt_gamma(self, var=None):
        # Unpack Parameters
        if var is None:
            var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        else:
            beta, c, lambda_ = var[:-2], var[-2:-1], var[-1:]
        anc1 = np.sqrt(self.delta) * beta.dot(beta.dot(self.invA))
        return (self.X.dot(beta) - c - 0.5 * self.regularization) / (lambda_ - anc1)
    
    def transport_X(self, var=None):
        # Unpack Parameters
        if var is None:
            var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        else:
            beta, c, lambda_ = var[:-2], var[-2:-1], var[-1:]
        opt_gamma = self.opt_gamma(var)
        res = self.X + np.transpose(np.sqrt(self.delta) * opt_gamma * beta.dot(self.invA))
        return res
    
    def _analytical_gradient(self):
        var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        transport_X = self.transport_X()
        opt_gamma = self.opt_gamma()
        grad_beta = np.transpose((2*transport_X.dot(beta) - 2*c - self.regularization) * transport_X.T)
        grad_c = - 2 * (transport_X.dot(beta) - c)
        grad_lambda = -np.sqrt(self.delta) * (np.square(opt_gamma)*beta.dot(beta.dot(self.invA)) - 1)
        return np.append(np.mean(grad_beta, axis = 0), [np.mean(grad_c), np.mean(grad_lambda)])
    
    def project(self):
        var, beta, c, lambda_ = self.var, self.beta, self.c, self.lambda_
        # Project beta
        sum_beta = np.sum(self.beta)
        beta += (1 - sum_beta) / len(beta)
        # Project lambda
        self.min_lambda = self.compute_min_lambda()
        if lambda_ < self.min_lambda + 1e-3:
            diff = self.min_lambda + 1e-3 - lambda_
            lambda_ += diff
    
    def plot_lambda_path(self):
        """
        Plot the path of lambda
        """
        x = range(1, self.total_iter + 1)
        plt.plot(x, self.var_path[:self.total_iter, -1])
        plt.plot(x, self.min_lambda_path[:self.total_iter])
        plt.title(self.model_name)
        plt.xlabel('Steps')
        plt.ylabel('Lambda')
        plt.show()
        
    def _check_dimension(self):
        # Check Dimension:
        if self.A.ndim not in (0,2,3):
            raise ValueError("The dimension of A should be 0, 2 or 3.")
        if self.X.ndim != 2:
            raise ValueError("The input data X should be a matrix")
        beta_dim = self.vardim - 2
        
        if self.A.ndim in (2,3):
            assert(beta_dim == self.A.shape[0] == self.A.shape[1] == self.X.shape[1])
        if self.A.ndim is 3:
            assert(self.num_train == self.A.shape[2])

