import copy
import math

import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    m = X.shape[0]
    total_cost = 0.
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        total_cost += -((y[i]) * np.log(f_wb) + (1 - y[i]) * np.log(1 - f_wb))

    total_cost /= m
    return total_cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        error = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, number_of_iterations, learning_rate):
    w_final = copy.deepcopy(w)
    b_final = b
    j_history = []
    for i in range(number_of_iterations):
        dj_dw, dj_db = compute_gradient(X, y, w_final, b_final)
        w_final -= learning_rate * dj_dw
        b_final -= learning_rate * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(compute_cost(X, y, w_final, b_final))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(number_of_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.20f}   ")
    return w_final, b_final, j_history


w_final, b_final, j_hist = gradient_descent(X_train, y_train, [1, 2], 3, 100000, 0.1)
print(w_final, b_final)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(j_hist[:100])
ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

