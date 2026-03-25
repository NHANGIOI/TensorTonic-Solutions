import numpy as np
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    for _ in range(0,steps):
        f_x = 2 * a * x0 + b
        x0 = x0 - lr * f_x
    return x0