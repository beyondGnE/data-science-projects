import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('../../datasets/moore.csv', header=None).values
X = data[:,0].reshape(-1, 1) # Convention is X is N x D matrix
Y = data[:,1]

plt.figure(dpi=300)
plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.figure(dpi=300)
plt.scatter(X, Y)
plt.show()

X2 = (X - X.mean()) # Avoid scaling input variable entirely
N, D = X.shape

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(D,)))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')

# Modify the learning rate over time.
def schedule(epoch, lr):
	if epoch >= 50:
		return 0.0001
	return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X2, Y, epochs=200, callbacks=[scheduler])

# Plot the loss
plt.figure(dpi=300)
plt.plot(r.history['loss'], label='loss')
plt.show()

# Input layer is like a "dummy" layer.
print(model.layers)
# Remember that w can be a 2-D matrix!
print(model.layers[0].get_weights())
print(model.layers[0].get_weights()[0][0][0])
print(model.layers[0].get_weights()[1][0])
m = model.layers[0].get_weights()[0][0][0]
b = model.layers[0].get_weights()[1][0]
plt.figure(dpi=300)
# plt.scatter(X, Y)
plt.plot(m*X + b)
plt.show()

X = np.array(X2).flatten()
Y = np.array(Y)
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
print(a, b)
print("Time to double:", np.log(2) / a)
