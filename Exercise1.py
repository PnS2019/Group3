import os as os

import numpy as np
from tensorflow.python.keras import backend as k

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def task1(x, y, z):
	a = k.placeholder(shape=(5,))
	b = k.placeholder(shape=(5,))
	c = k.placeholder(shape=(5,))

	output = a * a + b * b + c * c + 2 * b * c
	func = k.function(inputs=(a, b, c), outputs=(output,))

	return func((x, y, z))


def task2(a):
	x = k.placeholder(shape=(1,))
	# tanh = sinh / cosh = (e^x - e^-x) / (e^x + e^-x)
	tanh = (k.exp(x) - k.exp(-x)) / (k.exp(x) + k.exp(-x))
	tanh_d = k.gradients(loss=tanh, variables=[x])

	func = k.function(inputs=(x,), outputs=(tanh_d[0],))
	return func((a,))


def task3(a):
	w = k.ones(shape=(2,))
	b = k.ones(shape=(1,))
	x = k.placeholder(shape=(2,))

	z = w[0] * x[0] + w[1] + x[1] + b[0]
	f = 1 / (1 + k.exp(-z))
	func = k.function(inputs=(x,), outputs=(f,))
	return func((a,))


def task4(n):
	p = np.random.rand(n)
	a = np.diag(np.arange(n))
	return a * p


print("Task 1:")
print(task1(np.arange(5), np.arange(5), np.arange(5)))
print()

print("Task 2:")
print("tanh'(1): ", end='')
print(task2(np.array([1])))
print()

print("Task 3:")
print(task3(np.array([1, 1])))
print()

print("Task 4:")
print(task4(5))
print()
