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
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(FX, FY, np.log1p(np.abs(F)), cmap='viridis')
    ax2.set_title('|F(fx, fy)|')
    plt.show()

def plot_ifft(f, dx, dy):
    F, _, _ = compute_fft(f, dx, dy)
    img = compute_ifft(F, dx, dy).real
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

img = mpimg.imread('image.png')     
gray = rgb2gray(img)

step = 1
dx = 1
dy = 1
# F, fx, fy = compute_fft(gray[::step, ::step], dx, dy)
# plot_fft(F, fx, fy)
plot_ifft(gray[::step, ::step], dx, dy)