import numpy as np 
import matplotlib.pyplot as plt 

def circ(x, y):
    r = np.sqrt(np.square(x) + np.square(y))
    return np.select([r < 1, r == 1], [1, 1/2], default=0)

def circ_graph(): 
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = np.linspace(-3, 3, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot_surface(X, Y, circ(X, Y))
    plt.show()