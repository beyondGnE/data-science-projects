import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_data(url):
	if url.split('.')[-1] == 'xls':
		data = pd.read_excel(url)
	elif url.split('.')[-1] == 'csv':
		data = pd.read_csv(url)
	X = data.values
	N, D = X.shape
	X = X.std()*np.random.randn(N, D) + X.mean()
	T = np.zeros(X.shape[0], dtype=int)
	for i in range(X.shape[0]):
		if np.mean(X[i]) > 90:
			T[i] = 2
		elif np.mean(X[i,0]) <= 90 and np.mean(X[i,0]) > 20:
			T[i] = 1
		else:
			T[i] = 0
	K = T.max() + 1

	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

	X, T = shuffle(X, T)

	X_train = X[:-int(X.shape[0]*0.2)]
	X_test = X[-int(X.shape[0]*0.2):]
	T_train = T[:-int(T.shape[0]*0.2)]
	T_test = T[-int(T.shape[0]*0.2):]
	# T_ind, K = build_target_matrix(T)
	return X_train, X_test, T_train, T_test

def build_target_matrix(T):
	# Apparently every piece of information for this function needs to be an integer.
	K = int(T.max()+1)
	T = T.astype(np.int32)
	T_ind = np.zeros((T.shape[0], K)).astype(np.int32)
	for i in range(T.shape[0]):
		T_ind[i, T[i]] = 1
	return T_ind, K