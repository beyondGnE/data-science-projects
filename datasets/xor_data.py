import matplotlib.pyplot as plt
import numpy as np
def get_xor():
	X = np.zeros((200, 2))
	X[:50] = np.random.random((50, 2)) / 2 + 0.5 # quad 1
	X[50:100] = np.random.random((50, 2)) / 2 # quad 3
	X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # quad 2
	X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # quad 4
	Y = np.array([0]*100 + [1]*100)
	return X, Y

def xor_viz():
	X, Y = get_xor()
	plt.scatter(X[:,0], X[:,1], c=Y)
	plt.show()

if __name__ == '__main__':
	xor_viz()