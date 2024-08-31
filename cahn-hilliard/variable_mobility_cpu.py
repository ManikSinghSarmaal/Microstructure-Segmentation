import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Set device
device = torch.device('cpu')

# Simulation parameters
Nsteps = 100000
dt = 0.01 #
N = 128
L = 64 * np.pi
dx = L / N

# Initial conditions and variables
c_hat = torch.empty((N, N), dtype=torch.complex64, device=device)
c = torch.empty((Nsteps, N, N), dtype=torch.float32, device=device)

# Noise and initial concentration
noise = 0.1
c0 = 0.4
rng = np.random.default_rng(12345)
c[0] = c0 + torch.tensor(noise * rng.standard_normal(c[0].shape), dtype=torch.float32, device=device)

# Plot initial condition
plt.imshow(c[0].cpu().numpy(), cmap='RdBu_r')
plt.colorbar()
plt.title(f'$c_0={c0:.1f}$')
plt.savefig('cahn-hilliard-initial.png')
plt.show()

# Parameters for the Cahn-Hilliard equation
W = 2.0  # Example value for W, adjust as needed
psi = 0.1  
kappa = 0.5  # Gradient coefficient

# Fourier space variables
kx = ky = torch.fft.fftfreq(N, d=dx) * 2 * np.pi
Kx, Ky = torch.meshgrid(kx, kx, indexing='ij')
K = torch.stack((Kx, Ky)).to(device)
K2 = torch.sum(K * K, dim=0)

epsilon = 1e-6
K2_reg = K2 + epsilon
# Anti-aliasing factor
kcut = kx.max() * 2.0 / 3.0
dealias = (torch.abs(K[0]) < kcut) * (torch.abs(K[1]) < kcut)

# The bulk free energy density derivative f'(c) = W * d/dc[c^2 * (1-c)^2]
def dfdc(c):
    return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)

# The variable mobility function M(c) = 1 + ψW(c) where W(c) = c^3 * (10 - 15c + 6c^2)
def M(c):
    Wc = c**3 * (10 - 15*c + 6*c**2)
    return 1 + psi * Wc
cint = c[0].sum()

for i in tqdm(range(1, Nsteps)):
    g_c = dfdc(c[i-1])
    
    #steps are:
    # 1.Compute the Laplacian of  c : This gives us  del^2 c  in real space.
	# 2.Compute  g_c  and  2kappa_c_del^2 c : Then compute the difference.
    laplacian_c_hat = -K2 * c_hat
    # Compute 2κc∇²c (in real space)
    laplacian_c_real = torch.fft.ifftn(laplacian_c_hat).real
    kappa_term = 2 * kappa * c[i-1] * laplacian_c_real
    # Compute H(c) = ∇[g_c - 2κc∇²c] (in Fourier space)
    H_c = g_c - kappa_term
    H_c_hat = torch.fft.fftn(H_c)
    # Apply the dealiasing factor
    H_c_hat *= dealias
    #  H˜(c) = H˜x + H˜y
    # Decompose H_c_hat into its x and y components
    H_c_hat_x = 1j * Kx * H_c_hat
    H_c_hat_y = 1j * Ky * H_c_hat
    # print(H_c_hat_x)
    # print(H_c_hat_y)

    # H˜(c) into real space
    H_c = torch.fft.ifftn(H_c_hat).real
    M_current = M(c[i-1])
    M_H_c = M_current * H_c
    M_H_c_hat = torch.fft.fftn(M_H_c)
    
    # Step 6: Evaluate ˜c(k, t + ∆t)
    c_hat[:] = torch.fft.fftn(c[i-1])
    c_hat[:] = (c_hat - dt * K2 * M_H_c_hat) / (1 + dt * kappa * K2**2)
    c[i] = torch.fft.ifftn(c_hat).real
    error = torch.abs(c[i].sum() - cint) / cint

print('Final concentration mean = ', c[-1].mean().cpu().numpy())
print('Relative error = ', error.cpu().numpy())

plt.imshow(c[-1].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
plt.title(f'$c_0={c0:.1f}$')
plt.savefig('cahn-hilliard-c0-{:.1f}.png'.format(c0))
plt.show()
from matplotlib import animation
from matplotlib.animation import PillowWriter

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(c[0].cpu().numpy(), cmap='RdBu_r', vmin=0.0, vmax=1.0)
cb = fig.colorbar(im, ax=ax, label=r'$c(x,y)$', shrink=0.8)
tx = ax.text(400, 50, f't={(25 * 0 * dt):.0f}', bbox=dict(boxstyle="round", ec='white', fc='white'))
ax.set_title(r'$c_0=%.1f$' % c0)

def animate(i):
    im.set_data(c[25 * i].cpu().numpy())
    im.set_clim(0.0, 1.0)
    tx.set_text(f't={(25 * i * dt):.0f}')
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=Nsteps // 25, interval=50)
ani.save(f'ch-c0={c0}.gif', writer='pillow', fps=24, dpi=100)