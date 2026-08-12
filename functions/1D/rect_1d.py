import numpy as np

def rect_1d(x, a):
    return np.select([np.absolute(a * x) < 1/2, np.absolute(a * x) == 1/2], [1, 1/2], default=0)