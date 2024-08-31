import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
import torch
from matplotlib import animation
from matplotlib.animation import PillowWriter

device = torch.device('mps')

# Simulation parameters
Nsteps = 10000
dt = 0.1

N = 256
L = 64 * np.pi
dx = L / N

# Cahn-Hilliard parameters
W = 2.0
kappa = 0.5
noise = 0.1
c0 = 0.7

# Define the mobility function M(c)
def mobility(c):
    M_m = 1.0  # Mobility in matrix phase
    M_p = 0.1  # Mobility in precipitate phase
    return M_m + (M_p - M_m) * c*2 * (1 - c)*2

# Fourier space vectors
kx = ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
Kx, Ky = torch.meshgrid(kx, kx, indexing='ij')
Kx, Ky = Kx.to(device), Ky.to(device)
K = torch.stack((Kx, Ky)).to(device)
K2 = torch.sum(K * K, dim=0).to(device)

# Anti-aliasing filter (dealiasing)
kcut = kx.max() * 2.0 / 3.0  # The Nyquist mode
dealias = (torch.abs(K[0]) < kcut) * (torch.abs(K[1]) < kcut).to(device)

# Initial condition
rng = np.random.default_rng(12345)
c = torch.empty((Nsteps, N, N), dtype=torch.float32, device=device)
c[0] = c0 + torch.tensor(noise * rng.standard_normal(c[0].shape), dtype=torch.float32, device=device)

# Calculate average mobility A
M_m = 1.0
M_p = 0.1
A = 0.5 * (M_m + M_p)

# Compute the interfacial and bulk energy densities
def fbulk(c):
    return W * c

def dfdc(c):
    return 2 * W * (c * (1 - c)*2 - (1 - c) * c*2)

def finterf(c_hat):
    return kappa * torch.fft.ifftn(K2 * c_hat).real

# Function to compute the vector H in Fourier space
def compute_H(c):
    c_hat = torch.fft.fftn(c).to(device)
    gc_hat = torch.fft.fftn(dfdc(c)).to(device)
    Hx_hat = 1j * Kx * (gc_hat - 2 * kappa * K2 * c_hat).to(device)
    Hy_hat = 1j * Ky * (gc_hat - 2 * kappa * K2 * c_hat).to(device)
    return Hx_hat, Hy_hat

c_hat = torch.fft.fftn(c[0]).to(device)
cint = c[0].sum()

for i in tqdm(range(1, Nsteps)):
    M_c = mobility(c[i-1]).to(device)
    Hx_hat, Hy_hat = compute_H(c[i-1])

    # Update in Fourier space using the semi-implicit scheme
    c_hat = (c_hat + dt * (1j * (Kx * Hx_hat + Ky * Hy_hat) * M_c) * dealias) / (1 + dt * A * kappa * K2**2)

    c[i] = (torch.fft.ifftn(c_hat).real).to(device)  # Back to real space
    error = torch.abs(c[i].sum() - cint) / cint

    if error > 1e-6:
        print(f"Warning: Mass conservation error at step {i} is {error:.6e}")

# Plot the final concentration field
plt.imshow(c[-1].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
plt.title(r'$c_0=%.1f$' % c0)
plt.colorbar()
plt.savefig('cahn-hilliard-variable-mobility-final.png')
plt.show()

# Animation
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(c[0].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
cb = fig.colorbar(im, ax=ax, label=r'$c(x,y)$', shrink=0.8)
tx = ax.text(400, 50, f't={(25*0*dt):.0f}', bbox=dict(boxstyle="round", ec='white', fc='white'))
ax.set_title(r'$c_0=%.1f$' % c0)

def animate(i):
    im.set_data(c[25 * i].cpu().numpy())
    im.set_clim(0.0, 1.0)
    tx.set_text(f't={(25 * i * dt):.0f}')
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=Nsteps // 25, interval=50)
ani.save('ch-variable-mobility-c0=' + str(c0) + '.gif', writer='pillow', fps=24, dpi=100)