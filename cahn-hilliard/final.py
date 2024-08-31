import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Set device
device = torch.device('cpu')

# Simulation parameters
Nsteps = 10000
dt = 0.01
N = 256
L = 64 * np.pi
dx = L / N

# Initial conditions and variables
c = torch.full((N, N), 0.0, dtype=torch.float32, device=device)

# Parameters for sphere generation
num_spheres = 50
min_radius = 1
max_radius = 10
c0_inside = 0.7
c0_outside = 0.3
spacing = 1

# List to hold positions and radii of the spheres
sphere_params = []

# Generate random spheres
rng = np.random.default_rng(123456)

for _ in range(num_spheres):
    attempts = 0
    while attempts < 100:  # Limit attempts to avoid infinite loop
        radius = rng.integers(min_radius, max_radius)
        center_x = rng.integers(radius, N - radius)
        center_y = rng.integers(radius, N - radius)

        # Check overlap with existing spheres
        overlap = False
        for (cx, cy, r) in sphere_params:
            if np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2) < r + radius + spacing:
                overlap = True
                break

        if not overlap:
            sphere_params.append((center_x, center_y, radius))
            
            # Generate a grid of indices
            y, x = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
            mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
            
            # Place the sphere into the grid
            c[mask] = c0_inside
            break
        
        attempts += 1

# Set the rest of the grid to the outside concentration
c[c == 0] = c0_outside

# Plot initial condition
plt.imshow(c.cpu().numpy(), cmap='RdBu_r')
plt.colorbar()
plt.title(f'Initial Condition: Spheres with c0_inside={c0_inside} and c0_outside={c0_outside}')
plt.savefig('cahn-hilliard-initial-spheres.png')
plt.show()

# Parameters for the Cahn-Hilliard equation
W = 2.0
psi = 0.1
kappa = 0.5

# Fourier space variables
kx = ky = torch.fft.fftfreq(N, d=dx).to(device) * 2 * np.pi
Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
K = torch.stack((Kx, Ky))
K2 = torch.sum(K * K, dim=0)

epsilon = 1e-6
K2_reg = K2 + epsilon

kcut = kx.max() * 2.0 / 3.0
dealias = ((torch.abs(K[0]) < kcut) * (torch.abs(K[1]) < kcut))

def dfdc(c):
    return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)

def M(c):
    Wc = c**3 * (10 - 15*c + 6*c**2)
    return 1 + psi*Wc

cint = c.sum()

c_history = [c.clone()]

for _ in tqdm(range(1, Nsteps)):
    g_c = dfdc(c)
    
    c_hat = torch.fft.fftn(c)
    laplacian_c_hat = -K2 * c_hat
    
    laplacian_c_real = torch.fft.ifftn(laplacian_c_hat).real
    kappa_term = 2 * kappa * c * laplacian_c_real
    
    H_c = g_c - kappa_term
    H_c_hat = torch.fft.fftn(H_c)
    
    H_c_hat *= dealias
    
    H_c_real = torch.fft.ifftn(H_c_hat).real
    
    M_current = M(c)
    A = 0.5*(M_current.min() + M_current.max())
    M_H_c = M_current * H_c_real
    M_H_c_hat = torch.fft.fftn(M_H_c)
    
    c_hat = torch.fft.fftn(c)
    c_hat = (c_hat - dt * K2 * M_H_c_hat) / (1 + A*dt * kappa * K2**2)
    c = torch.fft.ifftn(c_hat).real
    
    error = torch.abs(c.sum() - cint) / cint
    c_history.append(c.clone())

print('Final concentration mean = ', c.mean().item())
print('Relative error = ', error.item())

# Plot the final concentration field
plt.imshow(c.cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
plt.title(f'$c_0={c0_outside:.1f}$')
plt.savefig(f'cahn-hilliard-c0-{c0_outside:.1f}.png')
plt.show()

# Create an animation of the concentration field evolution
from matplotlib import animation

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(c_history[0].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
cb = fig.colorbar(im, ax=ax, label=r'$c(x,y)$', shrink=0.8)
tx = ax.text(0.05, 0.95, f't=0', transform=ax.transAxes, bbox=dict(boxstyle="round", ec='white', fc='white'))
ax.set_title(r'$c_0=%.1f$' % c0_outside)

def animate(i):
    im.set_array(c_history[25*i].cpu().numpy())
    tx.set_text(f't={(25 * i * dt):.2f}')
    return im, tx

ani = animation.FuncAnimation(fig, animate, frames=len(c_history)//25, interval=50)
ani.save(f'ch-c0={c0_outside:.1f}.gif', writer='pillow', fps=24, dpi=100)
plt.close()