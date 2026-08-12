import numpy as np

def rect_2d(x, y): 
    f_x = np.select([np.absolute(x) < 1/2, np.absolute(x) == 1/2], [1, 1/2], default=0)
    f_y = np.select([np.absolute(y) < 1/2, np.absolute(y) == 1/2], [1, 1/2], default=0)
    return f_x * f_y