import numpy as np 
import matplotlib.pyplot as plt

def comb_graph():
    step = 1
    x_vals = np.linspace(-3, 3, 1000)
    comb = np.zeros_like(x_vals)
    for n in np.arange(-3, 4, step):
        comb[np.argmin(np.abs(x_vals - n))] = 1

    plt.stem(x_vals[comb == 1], comb[comb == 1], markerfmt='^')
    ax = plt.gca()

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.text(ax.get_xlim()[1] + 0.05, 0, 'x', va='center')
    ax.text(0, ax.get_ylim()[1] + 0.05, 'y', ha='center', va='bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()