import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Set device (MPS for Mac, CUDA for Nvidia, or 'cpu' if no GPU available)
device = torch.device('mps')

# Simulation parameters
Nsteps = 10000
dt = 0.001
N = 256
L = 64 * np.pi
dx = L / N

# Initial conditions and variables
c = torch.zeros((Nsteps, N, N), dtype=torch.float32, device=device)  # Initialize with zeros

# Parameters for sphere generation
num_spheres = 5  # Number of spheres
min_radius = 5
max_radius = 15
c0_inside = 0.8  # Concentration inside the spheres
c0_outside = 0.4  # Concentration outside the spheres

# Generate random spheres
rng = np.random.default_rng(12345)
for _ in range(num_spheres):
    radius = rng.integers(min_radius, max_radius)
    center_x = rng.integers(radius, N - radius)
    center_y = rng.integers(radius, N - radius)
    
    # Generate a grid of indices
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2

    # Place the sphere into the grid
    c[0, center_x - radius:center_x + radius, center_y - radius:center_y + radius][mask] = c0_inside

# Set the rest of the grid to the outside concentration
c[0][c[0] == 0] = c0_outside
c_hat = torch.empty((N, N), dtype=torch.complex64, device=device)

# Plot initial condition to verify
plt.imshow(c[0].cpu().numpy(), cmap='RdBu_r')
plt.colorbar()
plt.title(f'Initial Condition: Spheres with c0_inside={c0_inside} and c0_outside={c0_outside}')
plt.savefig('cahn-hilliard-initial-spheres.png')
plt.show()

# Now you can proceed with your Cahn-Hilliard evolution using this c[0] as the starting point.
# Parameters for the Cahn-Hilliard equation
W = 2.0  # Example value for W, adjust as needed
psi = 0.1  
kappa = 0.5  # Gradient coefficient


# Fourier space variables
kx = ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
Kx, Ky = torch.meshgrid(kx, kx, indexing='ij')
Kx, Ky = Kx.to(device), Ky.to(device)
K = torch.stack((Kx, Ky)).to(device)
K2 = torch.sum(K * K, dim=0).to(device)

epsilon = 1e-6
K2_reg = K2 + epsilon
# Anti-aliasing factor
kcut = kx.max() * 2.0 / 3.0
dealias = ((torch.abs(K[0]) < kcut) * (torch.abs(K[1]) < kcut)).to(device)

# The bulk free energy density derivative f'(c) = W * d/dc[c^2 * (1-c)^2]
def dfdc(c):
    return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)

# The variable mobility function M(c) = 1 + ψW(c) where W(c) = c^3 * (10 - 15c + 6c^2)
def M(c):
    Wc = c**3 * (10 - 15*c + 6*c**2)
    return 1

cint = c[0].sum()

for i in tqdm(range(1, Nsteps)):
    g_c = dfdc(c[i-1])
    
    # Step 1: Compute the Laplacian of c in Fourier space
    c_hat[:] = torch.fft.fftn(c[i-1]).to(device)
    laplacian_c_hat = (-K2 * c_hat).to(device)
    
    # Step 2: Compute 2κc∇²c in real space
    laplacian_c_real = torch.fft.ifftn(laplacian_c_hat).real.to(device)
    kappa_term = (2 * kappa * c[i-1] * laplacian_c_real).to(device)
    
    # Compute H(c) = ∇[g_c - 2κc∇²c] in Fourier space
    H_c = (g_c - kappa_term).to(device)
    H_c_hat = torch.fft.fftn(H_c).to(device)
    
    # Apply the dealiasing factor
    H_c_hat *= dealias
    
    # Decompose H_c_hat into its x and y components
    H_c_hat_x = (1j * Kx * H_c_hat).to(device)
    H_c_hat_y = (1j * Ky * H_c_hat).to(device)
    
    # Convert H_c_hat back to real space
    H_c_real = torch.fft.ifftn(H_c_hat).real.to(device)
    
    # Apply the mobility function
    M_current = M(c[i-1]).to(device)
    A = 0.5*(M_current.min() + M_current.max())
    M_H_c = (M_current * H_c_real).to(device)
    M_H_c_hat = torch.fft.fftn(M_H_c).to(device)
    
    # Update concentration field in Fourier space
    c_hat[:] = torch.fft.fftn(c[i-1]).to(device)
    c_hat[:] = (c_hat - dt * K2 * M_H_c_hat) / (1 + A*dt * kappa * K2**2)
    c[i] = torch.fft.ifftn(c_hat).real.to(device)
    
    # Calculate error
    error = torch.abs(c[i].sum() - cint) / cint

print('Final concentration mean = ', c[-1].mean().cpu().numpy())
print('Relative error = ', error.cpu().numpy())

# Plot the final concentration field
plt.imshow(c[-1].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
plt.title(f'$c_0={c0_outside:.1f}$')
plt.savefig('cahn-hilliard-c0-{:.1f}.png'.format(c0_outside))
plt.show()

# Create an animation of the concentration field evolution
from matplotlib import animation
from matplotlib.animation import PillowWriter

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(c[0].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
cb = fig.colorbar(im, ax=ax, label=r'$c(x,y)$', shrink=0.8)
tx = ax.text(400, 50, f't={(25 * 0 * dt):.0f}', bbox=dict(boxstyle="round", ec='white', fc='white'))
ax.set_title(r'$c_0=%.1f$' % c0_outside)

def animate(i):
    im.set_data(c[25 * i].cpu().numpy())
    im.set_clim(0.0, 1.0)
    tx.set_text(f't={(25 * i * dt):.0f}')
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=Nsteps // 25, interval=50)
ani.save(f'ch-c0={c0_outside}.gif', writer='pillow', fps=24, dpi=100)