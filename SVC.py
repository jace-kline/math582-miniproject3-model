from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy as np
import numpy.linalg as la
from cvxopt import matrix, solvers
import matplotlib
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class Kernel(object):
    """
    Definitions of some common kernel functions.
    Call each of these functions with their respective kernel parameters to obtain a function object that acts on two training data points x, y.
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f

    @staticmethod
    def rbf(gamma):
        def f(x, y):
            return (np.exp(-gamma*(np.linalg.norm(x - y))**2))
        return f


class SVC(object):
    """
    Our definition of an SVC Classifier Model.
    """
    # constructor
    def __init__(self, kernel=Kernel.linear(), C=1.0):
        self.kernel = kernel
        self.C = C

    # helper function to build kernel matrix
    def _build_k(self, X):
        """
        build_k generates a kernel to use inside of an SVM calculation
        X: Training data for our calculations
        kernel_type: Specifies the type of kernel to use: linear_kernel, polynomial_kernel, rbf_kernel
        poly_power: An optional parameter to define to what degree the polynomial should be calculated
        gamma: An optional parameter that defines how far the influence of a single training example reaches
        :return:
        """
        kernel = self.kernel
        N = X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            x_i = X[i]
            for j in range(N):
                x_j = X[j]

                K[i][j] = kernel(x_i, x_j)

        return K

    # solve the dual SVM problem with given training data
    # find alphas, w, and b for use in predictions
    def fit(self, X, y):
        """
        SVM will calculate the weight and bias using the SVM quadratic method (soft margin)
        X: Training data used for calculations 
        y: results of training data
        kernel_type: Specifies the type of kernel to use: linear_kernel, polynomial_kernel, rbf_kernel
        C: Trades off misclassification of training examples against simplicity of the decision surface
        :return: weight, bias, and alphas matrix

        Help used: https://stats.stackexchange.com/questions/23391/how-does-a-support-vector-machine-svm-work/353605#353605
        """
        # map member variables / methods to shorter aliases
        C = self.C
        kernel = self.kernel
        build_k = self._build_k

        # Grabs shape of our training data
        m, _ = X.shape

        # Make sure y values are floats and within -1 == y == 1
        y = y.reshape(-1,1) * 1.

        # Calculate our kernel
        K = build_k(X)

        # Compute 
        H = np.matmul(y,y.T) * K * 1.

        #Converting into cvxopt format - as previously
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        #Run solver
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # Calculating w, b
        w = ((y * alphas).T @ X).reshape(-1,1).flatten()
        S = (alphas > 1e-4).flatten()

        sv = X[S]
        sv_y = y[S]
        alphas = alphas[S]
        b = sv_y - np.sum(build_k(sv) * alphas * sv_y, axis=0)
        b = [np.sum(b) / b.size]

        # set the member variables for building the predict method
        self.alphas = alphas
        self.w = w
        self.b = b

        ### build the prediction function ###

        # classifies a single sample as +1 or -1
        def classify_sample(x):
            return (1 if np.inner(w, x) + b >= 0 else -1)

        # classifies multiple samples as +1 or -1 -> outputs array
        def classify_samples(X_test):
            return np.apply_along_axis(classify_sample, 1, X_test)

        def classify(X):
            if X.shape == w.shape:
                return classify_sample(X)
            elif len(X.shape) > 1 and X[0].shape == w.shape:
                return classify_samples(X)
            else:
                raise Exception("Invalid test data shape. Either input an array (single sample) or a 2d array (multiple samples).")

        self._predict = classify

    # predict the binary classifications {-1,1} of test sample arrays in X
    # outputs a vector (or value) of classifications in {-1,1}
    # requires that fit() was called previously
    def predict(self, X):
        # if data not fitted -> predict function not valid
        if self._predict is None:
            raise Exception("SVC Model must be fitted before prediction. Utilize fit() method with training data.")

        # otherwise, return the prediction results
        # can be given a single sample (shape equal to w)
        # or multiple samples (a 2d array with elements that have shape equal to w)
        return self._predict(X)