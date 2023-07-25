import numpy as np


def soft_max(x):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp)
    return x_exp / x_exp_sum


x = np.array([[i] for i in range(5)])

print(soft_max(x))
