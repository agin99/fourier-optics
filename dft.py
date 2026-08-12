import numpy as np
import matplotlib.pyplot as plt 

def dft(f, xs, freqs):
    dx = xs[1] - xs[0]
    F = np.zeros(len(freqs), dtype=complex)
    for k, f_x in enumerate(freqs):
        F[k] = np.sum(f * np.exp(-2j*np.pi * f_x * xs))
    return F * dx

def inverse_dft(F, freqs, xs_out):
    dfx = freqs[1] - freqs[0]
    f = np.zeros(len(xs_out), dtype=complex)
    for i, x in enumerate(xs_out):
        f[i] = np.sum(F * np.exp(+2j*np.pi * freqs * x)) * dfx
    return f

def derive_xs_freqs(dx, lx):
    xs = np.arange(-lx / 2, lx / 2, dx)
    dfx = 1/lx
    bx = 1/dx
    freqs = np.arange(-bx/2, bx/2, dfx)
    return xs, freqs

def nyquist_rate(dx, lx, thresh=0.99):
    dx = 1e-1
    xs, freqs = derive_xs_freqs(dx, lx)
    f = np.sinc(xs)
    power = np.abs(dft(f, xs, freqs))**2

    order = np.argsort(np.abs(freqs))
    frac = np.cumsum(power[order]) / power.sum()
    B = np.abs(freqs)[order][np.searchsorted(frac, thresh)]
    return 2 * B