import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def deriv_activ(function):
	if function == relu:
		return deriv_relu
	elif function == tanh:
		return deriv_tanh
	elif function == sigmoid:
		return deriv_sigmoid
	elif function == softmax:
		return deriv_softmax

class Layer(object):
	def __init__(self, num_nodes=0, input_shape=None, activation='Identity'):
		self.m2 = num_nodes
		self.m1 = 1
		if input_shape is not None:
			self.m1 = input_shape[0]
		self.activation = deriv_map[activation][0]
		self.deriv_activ = deriv_map[activation][1]
		# The Z from current layer is the result from the previous layer.
		# self.Z = 
		# self.delta = np.empty((1, self.m1))
		# self.deriv_w = np.empty((self.m1, self.m2))
	def forward(self, X):
		if self.activation != 'Identity':
			return self.activation(X.dot(self.weights) + self.biases)
		return X.dot(w) + b
	def initialize_weights(self, m1):
		self.m1 = m1
		self.weights = np.random.randn(self.m1, self.m2) 
		self.biases = np.zeros((self.m2,))
	def set_weights(self, w):
		self.weights = w
	def get_weights(self):
		return self.weights
	def get_biases(self):
		return self.biases
	def set_biases(self, b):
		self.biases = b
	def set_m1(self, m1):
		self.m1 = m1
	def get_m1(self):
		return self.m1
	def set_m2(self, m2):
		self.m2 = m2
	def get_m2(self):
		return self.m2
	def set_Z(self, Z):
		self.Z = Z
	def get_Z(self):
		return self.Z

	def set_delta(self, delta):
		self.delta = delta
	def get_delta(self):
		return self.delta

	def set_deriv_w(self, deriv_w):
		self.deriv_w = deriv_w

	def get_deriv_w(self):
		return self.deriv_w

	def set_deriv_b(self, deriv_b):
		self.deriv_b = deriv_b

	def get_deriv_b(self):
		return self.deriv_b

	def get_activation(self):
		return self.activation



class feedforward(object):
	def __init__(self):
		self.layers = []
		

	def add(self, layer):
		self.layers.append(layer)

	# Constructs the neural network
	# Create Z, w, and b for every layer past the first one.
	def compile(self):
		# Make sure to add in the first layer!
		self.layers[0].initialize_weights(self.layers[0].get_m1())
		for i in range(1, len(self.layers)):
			# Create Z, w, and b for every layer.
			self.layers[i].initialize_weights(self.layers[i-1].get_m2())

	def get_layers(self):
		return self.layers

	# Must require a previous Z multiplied with current w and b
	def predict(self, X):
		self.layers[0].set_Z(self.layers[0].forward(X))
		for i in range(1, len(self.layers)):
			self.layers[i].set_Z(self.layers[i].forward(self.layers[i-1].get_Z()))
		return self.layers[-1].get_Z()

	# DOOD! You never updated the biases!!!
	def fit(self, X, T, lr=0.0001, epochs=10000, mu=0.95):
		velocitiesw = [0]*len(self.layers)
		velocitiesb = [0]*len(self.layers)
		# Go backwards through the layers
		errors = []
		for j in range(epochs):
			Y = self.predict(X)

			# Fixed
			self.layers[-1].set_delta(Y-T)
			for i in range(-2, -len(self.layers)-1, -1):
				# print(i)
				# print(self.layers[i])
				self.layers[i].set_delta((self.layers[i+1].get_delta().dot(self.layers[i+1].get_weights().T))*self.layers[i].get_activation()(self.layers[i].get_Z()))
			# print([layer.get_delta().shape for layer in self.layers])
			# Fixed
			self.layers[0].set_deriv_w(X.T.dot(self.layers[0].get_delta()))
			self.layers[0].set_deriv_b(np.sum(self.layers[0].get_delta(), axis=0))
			for i in range(1, len(self.layers)):
				self.layers[i].set_deriv_w(self.layers[i-1].get_Z().T.dot(self.layers[i].get_delta()))
				self.layers[i].set_deriv_b(np.sum(self.layers[i].get_delta(), axis=0))

			# print([layer.get_deriv_w().shape for layer in self.layers])
			# print([layer.get_deriv_b().shape for layer in self.layers])

			for i in range(len(self.layers)):
				velocitiesw[i] = velocitiesw[i]*mu - lr*self.layers[i].get_deriv_w()
				velocitiesb[i] = velocitiesb[i]*mu - lr*self.layers[i].get_deriv_b()
			# Fixed
			for i in range(len(self.layers)):
				# self.layers[i].set_weights(self.layers[i].get_weights() - lr*self.layers[i].get_deriv_w())
				# self.layers[i].set_biases(self.layers[i].get_biases() - lr*self.layers[i].get_deriv_b())
				self.layers[i].set_weights(self.layers[i].get_weights() + mu*velocitiesw[i] - lr*self.layers[i].get_deriv_w())
				self.layers[i].set_biases(self.layers[i].get_biases() + mu*velocitiesb[i] - lr*self.layers[i].get_deriv_b())

			e = multi_cross_entropy(T, Y)
			errors.append(e)
			if j % 1000 == 0:
				print(j, e)
		plt.figure(dpi=300)
		plt.plot(errors)
		plt.show()

	def accuracy(self, T, Y):
		return np.mean(T == Y)
def main():
	# X = pd.read_excel('../../datasets/mlr02.xls').values
	X = donut()
	X = (X - X.mean()) / X.std()
	# T = np.array([0, 2, 1, 1, 0, 2, 1, 3, 1, 3, 1])
	T = np.array([0]*(X.shape[0]//2) + [1]*(X.shape[0]//2)).astype(np.int32)
	K = T.max() + 1
	T_mat = np.zeros((T.shape[0], K))
	for i in range(T.shape[0]):
		T_mat[i, T[i]] = 1

	print(T_mat)

	model = feedforward()
	# model.add(Layer(10, input_shape=(X.shape[1],), activation='tanh'))
	# model.add(Layer(50, activation='tanh'))
	# model.add(Layer(70, activation='tanh'))
	# model.add(Layer(30, activation='tanh'))
	# model.add(Layer(50, activation='tanh'))
	# model.add(Layer(20, activation='tanh'))
	# model.add(Layer(K, activation='softmax'))
	model.add(Layer(20, input_shape=(X.shape[1],), activation='relu'))
	# model.add(Layer(50, activation='relu'))
	# model.add(Layer(70, activation='relu'))
	# model.add(Layer(30, activation='relu'))
	# model.add(Layer(50, activation='relu'))
	# model.add(Layer(20, activation='relu'))
	model.add(Layer(K, activation='softmax'))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(600, activation=relu))
	# model.add(Layer(2, activation=softmax))
	model.compile()

	print([layer.get_weights().shape for layer in model.get_layers()])
	print([layer.get_biases().shape for layer in model.get_layers()])

	# Y = model.predict(X)

	# print([layer.get_Z().shape for layer in model.get_layers()])
	# print([layer.get_Z() for layer in model.get_layers()])
	
	# print([layer.get_weights().shape for layer in model.get_layers()])
	# print([layer.get_biases().shape for layer in model.get_layers()])
	# print([layer.get_weights() for layer in model.get_layers()])
	# print([layer.get_biases() for layer in model.get_layers()])

	Y = model.predict(X)

	print([layer.get_Z().shape for layer in model.get_layers()])
	# print([layer.get_Z() for layer in model.get_layers()])

	model.fit(X, T_mat)

	# print([layer.get_delta().shape for layer in model.get_layers()])
	# print([layer.get_delta() for layer in model.get_layers()])

	# print([layer.get_deriv_w().shape for layer in model.get_layers()])
	# print([layer.get_deriv_w() for layer in model.get_layers()])

	Y = model.predict(X)
	print("Y: ", np.argmax(Y, axis=1))
	print("T: ", T)

	print("Accuracy:", model.accuracy(T, np.argmax(Y, axis=1)))

	plt.figure(dpi=300)
	plt.title('Classified donut')
	plt.scatter(X[:,0], X[:,1], c=np.argmax(Y, axis=1))
	plt.show()
if __name__ == '__main__':
	main()

# Maybe...you just had a bad configuration?
