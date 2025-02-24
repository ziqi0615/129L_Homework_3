#!/usr/bin/env python

from numpy import exp, dot, pi, inf, isclose, array, sum
from scipy.linalg import cholesky, LinAlgError, solve_triangular, inv
from scipy.integrate import nquad

# a
def integral_function(vec, mat, vec_w, power=0):
	vec = array(vec)
	return exp(-dot(vec, dot(mat, vec))/2 + dot(vec, vec_w)) * (vec**power).prod()

# Using Cholesky decomposition is a bit faster than using inv directly.
def analytical_integral(mat, vec_w):
	chol_mat = cholesky(mat, lower=True)
	det_chol_mat = chol_mat.diagonal().prod()
	inv_chol_vec_w = solve_triangular(chol_mat, vec_w, lower=True)
	return (2*pi)**(len(vec_w)/2)/det_chol_mat*exp(dot(inv_chol_vec_w, inv_chol_vec_w)/2)

def numerical_integral(mat, vec_w):
	return nquad(
		lambda *args: integral_function(args[:-2], *args[-2:]),
		[(-inf, inf)]*len(vec_w),
		args=(mat, vec_w)
	)[0]

def validate_integral(mat, vec_w):
	try:
		expected = analytical_integral(mat, vec_w)
		computed = numerical_integral(mat, vec_w)
	except LinAlgError:
		print("error: not positive definite")
		return
	if isclose(expected, computed):
		print("pass")
	else:
		print(f"fail: analytic {expected}, numerical {computed}")

# b
MATRIX_A = [[4,2,1],[2,5,3],[1,3,6]]
VECTOR_W = [1,2,3]
validate_integral(MATRIX_A, VECTOR_W) # correct
validate_integral([[4,2,1],[2,1,3],[1,3,6]], VECTOR_W) # not positive definite

# c
# len(variables) must be even number.
# For each possible way of pairing elements in variables,
# multiply the corresponding elements in cov_matrix.
# Then sum all the products.
def pairwise_product(cov_matrix, variables):
	order = len(variables)
	if order == 0:
		return 1
	result = 0
	for i in range(1, order):
		result += cov_matrix[variables[0], variables[i]] * pairwise_product(cov_matrix, variables[1:i] + variables[i+1:])
	return result

# Compute moments for multivariate normal distribution with covariance cov_matrix and mean mean_vector
def wick_theorem(cov_matrix, mean_vector, variables, shifted=None):
	num_vars = len(variables)
	if num_vars == 0:
		return 1
	if shifted is None:
		shifted = [False]*num_vars
	result = 0
	for i in range(num_vars):
		if shifted[i]:
			continue
		shifted[i] = True
		result += mean_vector[variables[i]] * wick_theorem(cov_matrix, mean_vector, variables[:i] + variables[i+1:], shifted[:i] + shifted[i+1:])
	if num_vars % 2 == 0 and all(shifted):
		result += pairwise_product(cov_matrix, variables)
	return result

def analytical_moment(mat, vec_w, power_vec):
	cov_matrix = inv(mat)
	variables = []
	for idx, p in enumerate(power_vec):
		variables += [idx]*p
	return wick_theorem(cov_matrix, dot(cov_matrix, vec_w), variables)

def numerical_moment(mat, vec_w, power_vec):
	return nquad(
		lambda *args: integral_function(args[:-3], *args[-3:]),
		[(-inf, inf)]*len(vec_w),
		args=(mat, vec_w, power_vec)
	)[0] / numerical_integral(mat, vec_w)

def validate_moment(mat, vec_w, power_vec):
	expected = analytical_moment(mat, vec_w, power_vec)
	computed = numerical_moment(mat, vec_w, power_vec)
	if isclose(expected, computed):
		print("pass")
	else:
		print(f"fail: analytic {expected}, numerical {computed}")

# Validation for various moments
validate_moment(MATRIX_A, VECTOR_W, [1,0,0]) # mu1
validate_moment(MATRIX_A, VECTOR_W, [0,1,0]) # mu2
validate_moment(MATRIX_A, VECTOR_W, [0,0,1]) # mu3
validate_moment(MATRIX_A, VECTOR_W, [1,1,0]) # S12 + mu1 mu2
validate_moment(MATRIX_A, VECTOR_W, [0,1,1]) # S23 + mu2 mu3
validate_moment(MATRIX_A, VECTOR_W, [1,0,1]) # S13 + mu1 mu3
validate_moment(MATRIX_A, VECTOR_W, [2,1,0]) # 2 mu1 S12 + mu2 S11 + mu1^2 mu2
validate_moment(MATRIX_A, VECTOR_W, [0,1,2]) # 2 mu3 S23 + mu2 S33 + mu2 mu3^3
validate_moment(MATRIX_A, VECTOR_W, [2,2,0]) # (S11 + mu1^2) (S22 + mu2^2) + 2 S12^2
validate_moment(MATRIX_A, VECTOR_W, [0,2,2]) # (S22 + mu2^2) (S33 + mu3^2) + 2 S23^2
