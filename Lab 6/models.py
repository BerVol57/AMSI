import numpy as np

def model1(params, x):
    b1, b2 = params
    return b1 * (1 - np.exp(-b2 * x))

bounds1 = np.array([[0, 100], [0, 1]])

def model2(params, x):
    b1, b2, b3 = params
    return np.exp(-b1 * x) / (b2 + b3 * x)

bounds2 = np.array([[0, 1], [0, 1], [0, 1]])

def model3(params, x):
    b1, b2, b3, b4, b5, b6 = params
    return b1 * np.exp(-b2 * x) + b3 * np.exp(-b4 * x) + b5 * np.exp(-b6 * x)

bounds3 = np.array([[0, 1], [0, 1], [0, 1], 
                    [0, 10], [0, 10], [0, 10]])