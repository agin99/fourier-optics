import numpy as np
from scipy import fft 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def compute_fft(f, dx, dy):
    fx = fft.fftshift(fft.fftfreq(f.shape[1], d=dx))
    fy = fft.fftshift(fft.fftfreq(f.shape[0], d=dy))
    F  = fft.fftshift(fft.fft2(f)) * dx * dy

    return F, fx, fy

def compute_ifft(F, dx, dy): 
    f = fft.ifft2(fft.ifftshift(F)) / (dx * dy)

    return f

def plot_fft(F, fx, fy):
    FX, FY = np.meshgrid(fx, fy)
    fig = plt.figure(figsize=(12, 5))
    ax2 = fig.add_subplot(projection='3d')
    ax2.plot_surface(FX, FY, np.log1p(np.abs(F)), cmap='viridis')
    ax2.set_title('|F(fx, fy)|')
    plt.show()

def plot_ifft(f, dx, dy):
    F, _, _ = compute_fft(f, dx, dy)
    img = compute_ifft(F, dx, dy).real
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def compress_img(f, dx, dy, thresh=0.99):
    F, fx, fy = compute_fft(f, dx, dy)
    power = np.abs(F)**2
    FX, FY = np.meshgrid(fx, fy)
    r = np.sqrt(FX**2 + FY**2)

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    p_sorted = power.ravel()[order]

    frac = np.cumsum(p_sorted) / p_sorted.sum()
    r99 = r_sorted[np.searchsorted(frac, thresh)]

    mask = r <= r99
    F_compressed = F * mask
    return mask, compute_ifft(F_compressed, dx, dy).real

def power_spectrum_compression(img_gray, dx, dy):
    vmin, vmax = img_gray.min(), img_gray.max()
    thresh_list = [0.99, 0.9925, 0.995, 0.9975, 0.999, 0.9999, 0.99999]
    n = len(thresh_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), layout="constrained")
    for col, thresh in enumerate(thresh_list):
        mask, image_reconstruction = compress_img(img_gray, dx, dy, thresh=thresh)
        axes[col].imshow(image_reconstruction, cmap='gray', vmin=vmin, vmax=vmax)
        axes[col].set_title(f"{thresh:.4g} — {mask.sum()/mask.size:.1%} of coeffs")
        axes[col].axis('off')

    plt.show()

def mag_only_ifft(img_gray, dx, dy): 
    F, _, _ = compute_fft(img_gray, dx, dy)
    F_mag_only = np.abs(F).astype(complex)

    image_recon = compute_ifft(F_mag_only, dx, dy).real
    plt.imshow(np.log1p(np.abs(image_recon)), cmap='gray')
    plt.axis('off')
    plt.show()


def phase_only_ifft(img_gray, dx, dy): 
    F, _, _ = compute_fft(img_gray, dx, dy)
    F_phase_only = F / np.abs(F)

    image_recon = compute_ifft(F_phase_only, dx, dy).real
    plt.imshow(image_recon, cmap='gray')
    plt.axis('off')
    plt.show()
    

img = mpimg.imread('image.png')     
gray = rgb2gray(img)

step = 1
dx = 1
dy = 1

phase_only_ifft(gray[::step, ::step], dx, dy)