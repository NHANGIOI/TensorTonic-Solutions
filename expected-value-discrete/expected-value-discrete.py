import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x,dtype = np.float64)
    p = np.array(p,dtype = np.float64)
    if (abs(np.sum(p) - 1.0) > 1e-6):
        raise ValueError("Invalid probability distribution")
    else:
        return np.sum(x * p)
