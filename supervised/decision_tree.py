import numpy as np
from util import get_data, get_xor, get_donut
from datetime import datetime

def entropy(y):
	N = len(y) # Save the length of the target vector
	s1 = (y == 1).sum() # Sum all the elements of y that are equal to 1
	if 0 == s1 or N == s1: # if no elements of y are 1 or the length is 0
		return 0 
	p1 = float(s1) / N # Get the probability of s1 over N
	p0 = 1 - p1 # Get the opposing probability of p1
	return -p0*np.log2(p0) - p1*np.log2(p1) # The total entropy

class TreeNode:
	def __init__(self, depth=0, max_depth=None):
		self.depth = depth
		self.max_depth = max_depth

	def fit(self, X, Y):
		if len(Y) == 1 or len(set(Y)) == 1: # The first base case: Only one sample
			self.col = None
			self.split = None
			self.left = None
			self.right = None
			self.prediction = Y[0]
		else:
			D = X.shape[1]
			cols = range(D)

			max_ig = 0
			best_col = None
			best_split = None
			for col in cols:
				ig, split = self.find_split(X, Y, col)
				if ig > max_ig:
					max_ig = ig
					best_col = col
					best_split = split

			if max_ig == 0: # base case 2: No more splits to be made
				self.col = None
				self.split = None
				self.left = None
				self.right = None
				self.prediction = np.round(Y.mean())
			else:
				self.col = best_col
				self.split = best_split

				if self.depth == self.max_depth: # base case 3: reached max depth
					self.left = None
					self.right = None
					self.prediction = [
						np.round(Y[X[:,best_col] < self.split].mean()),
						np.round(Y[X[:,best_col] >= self.split].mean()),
					]
				else:
					left_idx = (X[:,best_col] < best_split)
					Xleft = X[left_idx]
					Yleft = Y[left_idx]
					self.left = TreeNode(self.depth + 1, self.max_depth)
					self.left.fit(Xleft, Yleft)

					right_idx = (X[:,best_col] >= best_split)
					Xright = X[right_idx]
					Yright = Y[right_idx]
					self.right = TreeNode(self.depth + 1, self.max_depth)
					self.right.fit(Xright, Yright)

	def find_split(self, X, Y, col):
		x_values = X[:,col]
		sort_idx = np.argsort(x_values)
		x_values = x_values[sort_idx]
		y_values = Y[sort_idx]

		boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
		best_split = None
		max_ig = 0
		# for i in boundaries:
		# 	split = (x_values[i] + x_values[i+1]) / 2
		for b in boundaries:
			split = (x_values[b] + x_values[b+1]) / 2
			ig = self.information_gain(x_values, y_values, split)
			if ig > max_ig:
				max_ig = ig
				best_split = split
		return max_ig, best_split

	def information_gain(self, x, y, split):
		y0 = y[x < split]
		y1 = y[x >= split]
		N = len(y)
		y0len = len(y0)
		if y0len == 0 or y0len == N:
			return 0
		p0 = float(len(y0)) / N
		p1 = 1 - p0
		return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

	def predict_one(self, x):
		if self.col is not None and self.split is not None:
			feature = x[self.col]
			if feature < self.split:
				if self.left:
					p = self.left.predict_one(x)
				else:
					p = self.prediction[0]
			else:
				if self.right:
					p = self.right.predict_one(x)
				else:
					p = self.prediction[1]
		else:
			p = self.prediction
		return p

	# def predict(self, x): # THIS IS THE ERROR! Upper casing! Fixing it now works.
	def predict(self, X):
		N = len(X) # Uses X of X[idx] instead of parameter!
		P = np.zeros(N)
		for i in range(N):
			P[i] = self.predict_one(X[i])
		return P

class DecisionTree: # A wrapper class
	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	def fit(self, X, Y):
		# print(X.shape)
		# print(Y.shape)
		self.root = TreeNode(max_depth=self.max_depth)
		self.root.fit(X, Y)
		print(X.shape)
		print(Y.shape)

	def predict(self, X):
		return self.root.predict(X)

	def score(self, X, Y):
		P = self.predict(X)
		print(P.shape)
		print(Y.shape)
		return np.mean(P == Y)

if __name__ == '__main__':
	X, Y = get_data()
	# X, Y = get_xor()
	# X, Y = get_donut()
	# print(X.shape)
	# print(Y.shape)

	idx = np.logical_or(Y == 0, Y == 1)
	X = X[idx]
	Y = Y[idx]
	
	# print(X.shape)
	# print(Y.shape)

	Ntrain = len(Y) // 2
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# print(Xtrain.shape)
	# print(Ytrain.shape)
	# print(Xtest.shape)
	# print(Ytest.shape)

	model = DecisionTree()
	t0 = datetime.now()
	model.fit(Xtrain, Ytrain)
	print("Training time:", (datetime.now() - t0))

	t0 = datetime.now()
	print("Train accuracy:", model.score(Xtrain, Ytrain))
	print("Time to compute train accuracy:", (datetime.now() - t0))

	t0 = datetime.now()
	print("Test accuracy:", model.score(Xtest, Ytest))
	print("Time to compute test accuracy:", (datetime.now() - t0))


