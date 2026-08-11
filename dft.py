import numpy as np
import matplotlib.pyplot as plt 

from functions.rect import rect_1d
from functions.sgn import sgn
from functions.tri import tri

def func(x): 
    return np.sin(x) + 2*np.sin(2*x) + 4 * np.sin(4*x)

def dft(f, xs, freqs):
    F = np.zeros(len(freqs), dtype=complex)
    for k, nu in enumerate(freqs):
        F[k] = np.sum(f * np.exp(-2j*np.pi * nu * xs))

    return F / len(xs)


def plot_transform(xs, freqs):
    scales = [0, 1, 4, 16, 64, 296]
    mask = np.abs(xs) < 5
    fig, axes = plt.subplots(len(scales), 2, figsize=(10, 12))

    for row, a in enumerate(scales):
        func_type = "sgn"
        f = np.where(a == 0, np.ones_like(xs), a * rect_1d(xs, a))

        title = "$g(x) = 1$" if a == 0 else f"${a}\\,\\mathrm{{{func_type}}}({a}x)$"

        axes[row, 0].plot(xs[mask], f[mask])
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel("$x$")
        axes[row, 0].set_ylabel("$g(x)$")

        axes[row, 1].plot(freqs, (1e2) * np.abs(dft(f, xs, freqs))) #L = 1e2 => L/N
        axes[row, 1].set_title(f"$|G|$, a={a}")
        axes[row, 1].set_ylim(0, 1.1)
        axes[row, 1].set_xlabel("$f_x$ (cycles / unit_length)")
        axes[row, 1].set_ylabel("$|G(f_x)|$")

    plt.tight_layout()
    plt.show()

spatial_step_size = 1e-3
spatial_region_l = 1e2
spatial_step_count = spatial_region_l / spatial_step_size
freq_range = 6
rect_scaling = 4
xs = np.arange(-spatial_region_l / 2, spatial_region_l / 2, spatial_step_size)
freqs = np.arange(-freq_range / 2, freq_range / 2, 1/spatial_region_l)

plot_transform(xs, freqs)
