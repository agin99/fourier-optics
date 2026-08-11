import numpy as np 
import matplotlib.pyplot as plt 

def sgn(x): 
    return np.select([x > 0, x < 0], [1, -1], default=0)

def sgn_graph():
    x_vals = np.linspace(-5, 5, 1000)
    plt.plot(x_vals, sgn(x_vals))

    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_linewidth(2)
    ax.text(0, ax.get_ylim()[1] + 0.05, ' y', ha='center', va='bottom')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_linewidth(2)
    ax.text(ax.get_xlim()[1] + 0.05, 0, ' x', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True)
    plt.show()