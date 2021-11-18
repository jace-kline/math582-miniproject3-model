from sklearn import svm
import numpy as np
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def polynomial_kernel (x_i, x_j, d):
	"""
	polynomial_kernel generates a kernel for SVM in a polynomial format

	x_i: ith column to chose from
	x_j: jth row to choose from 
	d: polynomial degree

	:return: kernel values for the i, jth value
	"""
	return (np.dot(x_i, x_j) + 1/2)**d

def linear_kernel (x_i, x_j):
	"""
	linear_kernel generates a kernel for SVM in a linear format

	x_i: ith column to chose from
	x_j: jth row to choose from 
	:return: kernel values for the i, jth value
	"""
	return (np.dot(x_i, x_j))

def rbf_kernel (x_i, x_j,gamma=1):
	"""
	rbf_kernel generates a kernel for SVM in a radial format

	x_i: ith column to chose from
	x_j: jth row to choose from 
	gamma: defines how far the influence of a single training example reaches
	:return: kernel values for the i, jth value
	"""
	return (np.exp(-gamma*(np.linalg.norm(x_j - x_i))**2))

def build_k (X, kernel_type='linear_kernel', poly_power=3, gamma=1):
	"""
	build_k generates a kernel to use inside of an SVM calculation
	X: Training data for our calculations
	kernel_type: Specifies the type of kernel to use: linear_kernel, polynomial_kernel, rbf_kernel
	poly_power: An optional parameter to define to what degree the polynomial should be calculated
	gamma: An optional parameter that defines how far the influence of a single training example reaches
	:return:
	"""
	N = X.shape[0]
	K = np.zeros((N, N))
	for i in range(X.shape[0]):
		x_i = X[i]
		for j in range(X.shape[0]):
			x_j = X[j]

			if kernel_type == 'linear_kernel':
				K[i][j] = linear_kernel(x_i, x_j)

			elif kernel_type == 'polynomial_kernel':
				K[i][j] = polynomial_kernel(x_i, x_j, poly_power)

			elif kernel_type == 'rbf_kernel':
				K[i][j] = rbf_kernel(x_i, x_j, gamma)

			else:
				raise ValueError('Use kernal type polynomial_kernel, linear_kernel or rbf_kernel') 

	return K

def SVM(X, y, kernel_type='linear_kernel', C=10):
	"""
	SVM will calculate the weight and bias using the SVM quadratic method (soft margin)
	X: Training data used for calculations 
	y: results of training data
	kernel_type: Specifies the type of kernel to use: linear_kernel, polynomial_kernel, rbf_kernel
	C: Trades off misclassification of training examples against simplicity of the decision surface
	:return: weight, bias, and alphas matrix

	Help used: https://stats.stackexchange.com/questions/23391/how-does-a-support-vector-machine-svm-work/353605#353605
	"""

	# Grabs shape of our training data
	m, _ = X.shape

	# Make sure y values are floats and within -1 == y == 1
    # This is the proper format that should be used in CVX opt
	y = y.reshape(-1,1) * 1.

	# Calculate our kernel
	K = build_k(X, kernel_type=kernel_type)

	# Compute H matrix for P
	H = np.matmul(y,y.T) * K * 1.

	# Converting into cvxopt format using their matrix library
    # All of this is just based off equations we discussed in class
	P = cvxopt_matrix(H)
	q = cvxopt_matrix(-np.ones((m, 1)))
	G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
	h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
	A = cvxopt_matrix(y.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))

	# Run solver using SVXopt quad prog
	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])

	# Calculating w, b
	w = ((y * alphas).T @ X).reshape(-1,1)
	S = (alphas > 1e-4).flatten()

	sv = X[S]
	sv_y = y[S]
	alphas = alphas[S]
	b = sv_y - np.sum(build_k(sv) * alphas * sv_y, axis=0)
	b = [np.sum(b) / b.size]

	return w, b, alphas


# Read in our training data from a CSV using pandas
df = pd.read_csv('./data/test-data/test_data.csv', encoding='utf8')
df["success"] = df[["success"]].replace(0,-1)

# Specify our X array by combining the training columns into a single 2D array.
X = df[['age', 'interest']]
# Grab the known y values
y = df[["success"]]

# Convert pandas data frame ---> numpy array
X = X.to_numpy()
y = y.to_numpy()

# Calculate our weight bias and alphas using our SVM function
w, b, alphas = SVM(X, y)

# Display results
print("------------------- FROM OUR CALCULATIONS -----------------------")
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b)

# Here, we look at the SVM calculations for a sanity check
print("------------------- FROM SVM CALCULATIONS -----------------------")
clf = SVC(C = 10, kernel = 'linear')
clf.fit(X, y.ravel()) 
w_svm=clf.coef_[0]
b_svm=clf.intercept_
print("w = ",w_svm) 
print("b = ",b_svm)

# Graph our resulting problem
# https://medium.com/geekculture/svm-classification-with-sklearn-svm-svc-how-to-plot-a-decision-boundary-with-margins-in-2d-space-7232cb3962c0
plt.figure(figsize=(8, 8))
# Constructing a hyperplane using a formula.
x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b[0] / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
colors = ["steelblue", "orange"]
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors), edgecolors="black", zorder=2,)
plt.plot(x_points, y_points, zorder=1, c='r');

plt.show()