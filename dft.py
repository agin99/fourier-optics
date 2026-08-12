import numpy as np
import matplotlib.pyplot as plt 

from functions.rect import rect_1d
from functions.sgn import sgn
from functions.tri import tri

def dft(f, xs, freqs):
    dx = xs[1] - xs[0]
    F = np.zeros(len(freqs), dtype=complex)
    for k, f_x in enumerate(freqs):
        F[k] = np.sum(f * np.exp(-2j*np.pi * f_x * xs))

    return F * dx

def inverse_dft(F, freqs, xs_out):
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

    order = np.argsort(np.abs(freqs))            # sort power & freqs together
    frac = np.cumsum(power[order]) / power.sum()
    B = np.abs(freqs)[order][np.searchsorted(frac, thresh)]
    return 2 * B

def plot_transform(dx, lx):
    scales = [0, 1, 4]
    fig, axes = plt.subplots(len(scales), 3, figsize=(10, 12))

    for row, a in enumerate(scales):
        func_type = "sinc"
        if a != 0: 
            dx /= a
        xs = np.arange(-lx / 2, lx / 2, dx)
        dfx = 1/lx
        bx = 1/dx
        freqs = np.arange(-bx/2, bx/2, dfx)
        f = np.where(a == 0, np.ones_like(xs), a * np.sinc(a * xs))

        mask = np.abs(xs) < 5
        title = "$g(x) = 1$" if a == 0 else f"${a}\\,\\mathrm{{{func_type}}}({a}x)$"

        xs_fine = np.arange(-lx/2, lx/2, 1e-2)
        truth_fine = np.ones_like(xs_fine) if a == 0 else a*np.sinc(a*xs_fine)
        mask_fine = np.abs(xs_fine) < 5
        F = dft(f, xs, freqs)
        recon = inverse_dft(F, freqs, xs_fine).real
        err = np.max(np.abs(recon - truth_fine)[np.abs(xs_fine) < lx/4])

        axes[row, 0].plot(xs_fine[mask_fine], a*np.sinc(a*xs_fine[mask_fine]), 'k-', lw=0.8)
        axes[row, 0].plot(xs[mask], f[mask], 'o', ms=4)
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel("$x$")
        axes[row, 0].set_ylabel("$g(x)$")

        axes[row, 1].plot(freqs, np.abs(F))
        axes[row, 1].set_title(f"$|G|$, a={a}")
        axes[row, 1].set_ylim(0, 1.1)
        axes[row, 1].set_xlabel("$f_x$ (cycles / unit_length)")
        axes[row, 1].set_ylabel("$|G(f_x)|$")

        axes[row, 2].plot(xs_fine[mask_fine], recon[mask_fine])
        axes[row, 2].set_title(title)
        axes[row, 2].set_xlabel("$x$")
        axes[row, 2].set_ylabel("$g(x)$")
        axes[row, 2].plot(xs_fine[mask_fine], truth_fine[mask_fine], 'k-', lw=0.8)
        axes[row, 2].set_title(f"recon, err={err:.2e}")

        if a != 0: 
            dx *= a

    plt.tight_layout()
    plt.show()

def sampling_example(dx, lx):
    nyq_scaling = [1/2, 0.97, 1, 3/2, 2]
    nyq_spacing = 1 / nyquist_rate(dx, lx)
    fig, axes = plt.subplots(len(nyq_scaling), 3, figsize=(10, 14), layout="constrained")
    fig.suptitle(f"Estimated Nyquist Rate: {nyq_spacing:.3f}   (analytic: {2*0.5:.1f})", fontsize=14)

    for row, a in enumerate(nyq_scaling):
        _dx = a * nyq_spacing
        xs_neg = np.arange(0, -lx / 2, -_dx)[1:]
        xs_pos = np.arange(0, lx / 2, _dx)
        xs = np.concatenate((np.flip(xs_neg), xs_pos))
        dfx = 1/lx
        bx = 1/_dx
        freqs = np.arange(-bx/2, bx/2, dfx)

        mask = np.abs(xs) < 5
        title = f"Spacing: {_dx:.4f}"

        xs_fine_neg = np.arange(0, -lx / 2, -1e-2)[1:]
        xs_fine_pos = np.arange(0, lx / 2, 1e-2)
        xs_fine = np.concatenate((np.flip(xs_fine_neg), xs_fine_pos))
        truth_fine = np.sinc(xs_fine)
        mask_fine = np.abs(xs_fine) < 5
        f = np.sinc(xs)
        f_fine = np.sinc(xs_fine)
        F = dft(f, xs, freqs)
        recon = inverse_dft(F, freqs, xs_fine).real

        axes[row, 0].plot(xs_fine[mask_fine], f_fine[mask_fine], 'k-', lw=0.8)
        axes[row, 0].plot(xs[mask], f[mask], 'o', ms=4)
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel("$x$")
        axes[row, 0].set_ylabel("$g(x)$")

        F_XLIM = 1.1
        axes[row, 1].plot(freqs, np.abs(F))
        axes[row, 1].set_xlim(-F_XLIM, F_XLIM)
        axes[row, 1].axvline(-0.5, color='gray', ls=':', lw=0.8)
        axes[row, 1].axvline(+0.5, color='gray', ls=':', lw=0.8)
        axes[row, 1].axvspan(-1/(2*_dx), 1/(2*_dx), alpha=0.08)
        axes[row, 1].set_title(f"$|G|$")
        axes[row, 1].set_xlabel("$f_x$ (cycles / unit_length)")
        axes[row, 1].set_ylabel("$|G(f_x)|$")

        axes[row, 2].plot(xs_fine[mask_fine], recon[mask_fine])
        axes[row, 2].set_xlabel("$x$")
        axes[row, 2].set_ylabel("$g(x)$")
        axes[row, 2].plot(xs_fine[mask_fine], truth_fine[mask_fine], 'k-', lw=0.8)
    
    plt.show()


dx = 5 / 2
lx = 1e2
xs = np.arange(-lx / 2, lx / 2, dx)
dfx = 1/lx
bx = 1/dx
freqs = np.arange(-bx/2, bx/2, dfx)
N = len(xs)
# assert len(freqs) == N
# assert np.isclose(dx * dfx, 1/N) #Check space-bandwidth product
# assert np.allclose(inverse_dft(dft(np.sinc(xs), xs, freqs), freqs, xs).real, np.sinc(xs), atol=1e-10)

# plot_transform(dx, lx)
sampling_example(dx, lx)