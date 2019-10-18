import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N, D = (1000, 2)

r1 = 5
r2 = 10

theta = 2*np.pi*np.random.randn(N)

X1 = np.random.randn(N, D) + r1
X2 = np.random.randn(N, D) + r2

X1[:,0] = X1[:,0]*np.cos(theta)
X1[:,1] = X1[:,1]*np.sin(theta)
X2[:,0] = X2[:,0]*np.cos(theta)
X2[:,1] = X2[:,1]*np.sin(theta)

X = np.concatenate((X1, X2))

# plt.scatter(X1[:,0], X1[:,1])
# plt.scatter(X2[:,0], X2[:,1])
plt.figure(dpi=300)
plt.scatter(X[:,0], X[:,1])
plt.show()
