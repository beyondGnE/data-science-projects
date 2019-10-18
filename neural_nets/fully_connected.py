import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def relu(a):
	return a * (a > 0)

def sigmoid(a):
	return 1 / (1 + np.exp(-a))

def softmax(a):
	return np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)

def tanh(a):
	return np.tanh(a)

def multi_cross_entropy(T, Y):
	return -np.mean(T*np.log(Y))

def deriv_relu(a):
	return a > 0

def deriv_tanh(a):
	return 1 - a*a

def deriv_sigmoid(a):
	return a*(1-a)

def deriv_softmax(a):
	return a

deriv_map = {
			'relu': [relu, deriv_relu],
			'tanh': [tanh, deriv_tanh],
			'sigmoid': [sigmoid, deriv_sigmoid],
			'softmax': [softmax, deriv_softmax],
			'Identity': None
		}

def donut():
	N = 1000
	D = 2

	R_inner = 5
	R_outer = 10

	R1 = np.random.randn(N//2) + R_inner
	theta = 2*np.pi*np.random.random(N//2)
	X_inner = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T

	R2 = np.random.randn(N//2) + R_outer
	theta = 2*np.pi*np.random.random(N//2)
	X_outer = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

	X = np.concatenate([ X_inner, X_outer ])
	return X

class Layer(object):
	def __init__(self, units=1, input_shape=None, activation='Identity'):
		self.m2 = units
		self.m1 = 1	
		if input_shape is not None:
			self.m1 = input_shape[0]
		self.activation = deriv_map[activation][0]
		self.deriv_activ = deriv_map[activation][1]

	def initialize_weightsnbiases(self, m1):
		# self.m1 = m1
		self.weights = np.random.randn(m1, self.m2)
		self.biases =  np.zeros((self.m2,))

	def forward(self, Z):
		return self.activation(Z.dot(self.weights) + self.biases)

	def update_weightsnbiases(self, lr, deriv_W, deriv_b):
		self.weights -= lr*deriv_W
		self.biases -= lr*deriv_b

	def get_m1(self):
		return self.m1

	def get_m2(self):
		return self.m2

	def get_weights(self):
		return self.weights

	def get_biases(self):
		return self.biases

	def get_deriv_activ(self):
		return self.deriv_activ

class fully_connected(object):
	def __init__(self):
		self.layers = []
		self.errors = []

	def add(self, layer):
		self.layers.append(layer)

	# Constructs the neural network
	# Create Z, w, and b for every layer past the first one.
	def compile(self):
		self.layers[0].initialize_weightsnbiases(self.layers[0].get_m1())
		for i in range(1, len(self.layers)):
			self.layers[i].initialize_weightsnbiases(self.layers[i-1].get_m2())
		print([layer.get_weights().shape for layer in self.layers])
		print([layer.get_biases().shape for layer in self.layers])

	# Must require a previous Z multiplied with current w and b
	def predict(self, X):
		self.Zs = []		# Z is reborn for every prediction
		self.Zs.append(self.layers[0].forward(X))
		for i in range(1, len(self.layers)):
			self.Zs.append(self.layers[i].forward(self.Zs[i-1]))
		return self.Zs[-1]

	def find_deltas(self, X, Zs, Y, T):
		self.deltas = []
		self.deltas.append(Y-T)
		for i in range(1, len(self.layers)):
			self.deltas.append(self.deltas[i-1].dot(self.layers[-1+1-i].get_weights().T)*self.layers[-2+1-i].get_deriv_activ()(Zs[-2+1-i]))

	def find_derivs(self, X):
		self.deriv_W = []
		self.deriv_b = []
		self.deriv_W.append(X.T.dot(self.deltas[-1]))
		self.deriv_b.append(np.sum(self.deltas[-1], axis=0))
		for i in range(1, len(self.layers)):
			self.deriv_W.append(self.Zs[i-1].T.dot(self.deltas[-1-i])) # Because the last Z is actually the Y, so it can be ignored
			self.deriv_b.append(np.sum(self.deltas[-1-i], axis=0))

	def update_equations(self, lr):
		for i in range(len(self.layers)):
			self.layers[i].update_weightsnbiases(lr, self.deriv_W[i], self.deriv_b[i])

	# Update the weights at every layer.
	def fit(self, X, T, epochs=10000, lr=0.0001):
		for i in range(epochs):
			Y = self.predict(X)
			e = multi_cross_entropy(T, Y)
			self.errors.append(e)
			self.find_deltas(X, self.Zs, Y, T)
			self.find_derivs(X)
			self.update_equations(lr)
			if i % 1000 == 0:
				print(i, e)

	def plot_errors(self):
		plt.figure(dpi=300)
		plt.plot(self.errors)
		plt.show()

	def accuracy(self, T, Y):
		print("Accuracy:", np.mean(T == np.argmax(Y, axis=1)))

	def get_layers(self):
		return self.layers

	def get_Zs(self):
		return self.Zs

	def get_deltas(self):
		return self.deltas

	def get_deriv_W(self):
		return self.deriv_W

	def get_deriv_b(self):
		return self.deriv_b

def main():
	# X = pd.read_excel('../../datasets/mlr02.xls').values
	X = donut()
	X = (X - X.mean()) / X.std()
	N, D = X.shape
	# T = np.array([0, 2, 1, 1, 0, 2, 1, 3, 1, 3, 1])
	T = np.array([0]*(X.shape[0]//2) + [1]*(X.shape[0]//2)).astype(np.int32)
	K = T.max() + 1
	T_mat = np.zeros((T.shape[0], K))
	for i in range(T.shape[0]):
		T_mat[i, T[i]] = 1

	X, T_mat = shuffle(X, T_mat)

	x_train = X[:int(N * 0.8)]
	y_train = T_mat[:int(N * 0.8)]
	x_test = X[int(N*0.8):]
	y_test = T_mat[int(N*0.8):]

	print(T_mat)

	# Could THIS be the optimal configuration for the donut problem?
	model = fully_connected()
	model.add(Layer(20, input_shape=(D,), activation='relu'))
	model.add(Layer(K, activation='softmax'))
	model.compile()
	model.fit(x_train, y_train)
	model.plot_errors()
	# Y = model.predict(X)
	# print([Z.shape for Z in model.get_Zs()])
	# model.find_deltas(X, model.get_Zs(), Y, T_mat)
	# print([delta.shape for delta in model.get_deltas()])
	# model.find_derivs(X)
	# print([dW.shape for dW in model.get_deriv_W()])
	# print([db.shape for db in model.get_deriv_b()])
	# model.fit()
	# model.evaluate()
	Y = model.predict(x_train)
	
	model.accuracy(np.argmax(y_train, axis=1), Y)
	print("T:",np.argmax(y_train, axis=1))
	print("Y:",np.argmax(Y, axis=1))

	# Test values:
	Y2 = model.predict(x_test)
	model.accuracy(np.argmax(y_test, axis=1), Y2)
	print("T:",np.argmax(y_test, axis=1))
	print("Y:",np.argmax(Y2, axis=1))

	Y_combined = np.concatenate([Y, Y2])
	plt.figure(dpi=300)
	plt.title('Classified donut')
	plt.scatter(X[:,0], X[:,1], c=np.argmax(Y_combined, axis=1))
	plt.show()

	X = pd.read_excel('../../datasets/mlr02.xls').values
	# X = donut()
	X = (X - X.mean()) / X.std()
	N, D = X.shape
	T = np.array([0, 2, 1, 1, 0, 2, 1, 3, 1, 3, 1])
	# T = np.array([0]*(X.shape[0]//2) + [1]*(X.shape[0]//2)).astype(np.int32)
	K = T.max() + 1
	T_mat = np.zeros((T.shape[0], K))
	for i in range(T.shape[0]):
		T_mat[i, T[i]] = 1

	X, T_mat = shuffle(X, T_mat)

	x_train = X[:int(N * 0.8)]
	y_train = T_mat[:int(N * 0.8)]
	x_test = X[int(N*0.8):]
	y_test = T_mat[int(N*0.8):]

	print(T_mat)

	model2 = fully_connected()
	model2.add(Layer(20, input_shape=(D,), activation='tanh'))
	model2.add(Layer(50, activation='tanh'))
	model2.add(Layer(70, activation='tanh'))
	model2.add(Layer(30, activation='tanh'))
	model2.add(Layer(50, activation='tanh'))
	model2.add(Layer(20, activation='tanh'))
	model2.add(Layer(K, activation='softmax'))
	model2.compile()
	model2.fit(x_train, y_train)
	model2.plot_errors()
	Y = model2.predict(x_train)
	model2.accuracy(np.argmax(y_train, axis=1), Y)
	print("T:",np.argmax(y_train, axis=1))
	print("Y:",np.argmax(Y, axis=1))
	# Test values:
	model2.fit(x_test, y_test)
	Y = model2.predict(x_test)
	model2.accuracy(np.argmax(y_test, axis=1), Y)
	print("T:",np.argmax(y_test, axis=1))
	print("Y:",np.argmax(Y, axis=1))
	
if __name__ == '__main__':
	main()

# NOW TO FIGURE OUT WHAT WENT WRONG BETWEEN THIS IMPLEMENTATION AND THE PREVIOUS ONE!
