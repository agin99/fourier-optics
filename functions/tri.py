import numpy as np
import matplotlib.pyplot as plt 

def tri(x): 
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

def tri_graph():
    x_vals = np.linspace(-5, 5, 1000)
    plt.plot(x_vals, tri(x_vals))

    ax = plt.gca() 

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(1.1)
    ax.spines['left'].set_position('zero')
    ax.text(0, ax.get_ylim()[1] + 0.05, 'y', ha='center', va='center')

    ax.spines['top'].set_linewidth(1.1)
    ax.spines['top'].set_position('zero')
    ax.text(ax.get_xlim()[1] + 0.05, 0, 'x', ha='center')

    plt.grid(True)
    plt.show()