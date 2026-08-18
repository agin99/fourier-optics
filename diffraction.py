import numpy as np
import matplotlib.pyplot as plt

def plane_wave(A: float, N: int):
    return A * np.ones((N, N))

def circ_aperture(D: float, dx: float, N: int):
    grid = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(grid, grid)
    R = X**2 + Y**2
    mask = R <= (D/2) ** 2

    return mask.astype(int)

def rect_aperture(w: float, d: float, dx: float, N: int): 
    grid = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(grid, grid)
    mask = (np.abs(X) <= w/2) & (np.abs(Y) <= d/2)

    return mask.astype(int)

# Angular Spectrum
def propagation_as(U0, lam, z, dx):
    N = U0.shape[0]
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(2j*np.pi*z/lam * np.sqrt((1 - (lam*FX)**2 - (lam*FY)**2).astype(complex)))
    A0 = np.fft.fft2(U0)
    Uz = np.fft.ifft2(A0 * H)

    return Uz

# First Rayleigh-Sommerfeld
def propagation_rs1(U0, lam, z, dx): 
    N = U0.shape[0]
    grid = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(grid, grid)
    R = np.sqrt(X**2 + Y**2 + z**2)
    k = 2*np.pi/lam
    h = z/(2*np.pi*R)*(1/R - 1j*k)*np.exp(1j*k*R)/R * dx**2
    Uz = np.fft.ifft2(np.fft.fft2(U0) * np.fft.fft2(np.fft.ifftshift(h)))

    return Uz

A = 1           # wave amplitude
N = 1024        # pixel count
D = 5e-6        # aperture diam
w = 5e-6        # aperture width
d = 5e-5        # aperture depth
dx = D / 100    # resolution
lam = 600e-9    # wavelength
z = 1e-6        # propagation distance

U0 = plane_wave(A, N) * circ_aperture(D, dx, N)
# U0 = plane_wave(A, N) * rect_aperture(w, d, dx, N)

# Uz = propagation_as(U0, lam, z, dx)
Uz = propagation_rs1(U0, lam, z, dx)
I = np.abs(Uz)**2

N = Uz.shape[0]
half = N/2 * dx * 1e6
extent = [-half, half, -half, half]

fig, ax = plt.subplots()
im = ax.imshow(I, extent=extent, cmap='inferno', origin='lower')
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_title(f'Intensity at z = {z*1e6:.0f} µm')
fig.colorbar(im, ax=ax, label='Intensity')
plt.show()