import numpy as np
import matplotlib.pyplot as plt 

def tri(x): 
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)