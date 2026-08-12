import numpy as np 
import matplotlib.pyplot as plt 

def sgn(x): 
    return np.select([x > 0, x < 0], [1, -1], default=0)