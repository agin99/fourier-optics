import numpy as np
import matplotlib.pyplot as plt

def rect_1d(x, a):
    return np.select([np.absolute(a * x) < 1/2, np.absolute(a * x) == 1/2], [1, 1/2], default=0)

def rect_1D_graph():
    x_vals = np.linspace(-5, 5, 1000)
    plt.plot(x_vals, rect_1d(x_vals))
    plt.show()

def rect_2d(x, y): 
    f_x = np.select([np.absolute(x) < 1/2, np.absolute(x) == 1/2], [1, 1/2], default=0)
    f_y = np.select([np.absolute(y) < 1/2, np.absolute(y) == 1/2], [1, 1/2], default=0)
    return f_x * f_y

def rect_2D_graph():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_vals = np.linspace(-5, 5, 1000)
    y_vals = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rect_2d(X, Y)

    ax.plot_surface(X, Y, Z)
    plt.show()
