import numpy as np
import matplotlib.pyplot as plt 

from dft import dft, inverse_dft, nyquist_rate

def generate_sym_domain(step_l, region_l):
    neg = np.arange(0, -region_l / 2, -step_l)[1:]
    pos = np.arange(0, region_l / 2, step_l)
    return np.concatenate((np.flip(neg), pos))

def plot_transform(dx, lx):
    OVERSAMPLE = 4
    dx /= OVERSAMPLE
    scales = [0, 1, 4]
    fig, axes = plt.subplots(len(scales), 3, figsize=(10, 12))

    for row, a in enumerate(scales):
        func_type = "sinc"
        if a != 0: 
            dx /= a
        
        xs = generate_sym_domain(dx, lx)
        mask = np.abs(xs) < 5
        f = np.where(a == 0, np.ones_like(xs), a * np.sinc(a * xs))

        xs_fine = generate_sym_domain(1e-2, lx)
        mask_fine = np.abs(xs_fine) < 5
        f_fine = np.where(a == 0, np.ones_like(xs_fine), a * np.sinc(a * xs_fine))

        dfx = 1 / (len(xs) * dx)
        bx = 1/dx
        freqs = generate_sym_domain(dfx, bx)
        decon = dft(f, xs, freqs)
        recon = inverse_dft(decon, freqs, xs_fine).real

        title = "$g(x) = 1$" if a == 0 else f"${a}\\,\\mathrm{{{func_type}}}({a}x)$"

        axes[row, 0].plot(xs[mask], f[mask], 'o', ms=4)
        axes[row, 0].plot(xs_fine[mask_fine], f_fine[mask_fine], 'k-', lw=0.8)
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel("$x$")
        axes[row, 0].set_ylabel("$g(x)$")

        F_XLIM = a * 0.5 + 1
        axes[row, 1].plot(freqs, np.abs(decon))
        axes[row, 1].set_xlim(-F_XLIM, F_XLIM)
        axes[row, 1].set_title(f"$|G|$, a={a}")
        axes[row, 1].set_xlabel("$f_x$ (cycles / unit_length)")
        axes[row, 1].set_ylabel("$|G(f_x)|$")

        axes[row, 2].plot(xs_fine[mask_fine], recon[mask_fine])
        axes[row, 2].plot(xs_fine[mask_fine], f_fine[mask_fine], 'k-', lw=0.8)
        axes[row, 2].set_title(title)
        axes[row, 2].set_xlabel("$x$")
        axes[row, 2].set_ylabel("$g(x)$")

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
        xs = generate_sym_domain(_dx, lx)
        xs_fine = generate_sym_domain(1e-2, lx)
        dfx = 1/lx
        bx = 1/_dx
        freqs = generate_sym_domain(dfx, bx)

        mask = np.abs(xs) < 5
        mask_fine = np.abs(xs_fine) < 5

        f = np.sinc(xs)
        f_fine = np.sinc(xs_fine)
        F = dft(f, xs, freqs)
        recon = inverse_dft(F, freqs, xs_fine).real

        title = f"Spacing: {_dx:.4f}"

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
        axes[row, 2].plot(xs_fine[mask_fine], f_fine[mask_fine], 'k-', lw=0.8)
    
    plt.show()

dx = 1
lx = 1e2
# plot_transform(dx, lx)
sampling_example(dx, lx)