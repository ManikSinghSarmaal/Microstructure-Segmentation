import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from scipy.fft import fft2, ifft2   # library for fourier transform

Nsteps = 10000  #Number of steps in the simulation
dt = 0.1

N = 256 # Grid size

# Arrays to hold Fourier transforms of the concentration field and its derivative.
c_hat = np.empty((N,N), dtype=np.complex64)
dfdc_hat = np.empty((N,N), dtype=np.complex64)

# Array to hold the concentration field over time
c = np.empty((Nsteps,N,N), dtype=np.float32)

dx = 1.0    # Spatial Step Size(distance between grid points)
L = N*dx    # Physical length of domain

#change initial conditions using spheres in the start
#Initial Conditions
noise = 0.1 #Random noise added to the initial concentration
c0 = 0.5    #Average concentration value

rng = np.random.default_rng(12345) # the seed of random numbers generator

c[0] = c0 + noise*rng.standard_normal(c[0].shape)   #Initial concentration field with added noise

# plt.imshow(c)
# plt.colorbar(cmap='RdBu_r')
# # plt.title('$c_0=%.1f$'% c0)
# plt.savefig('cahn-hilliard-input.png')
# plt.show()

print('c0 = ',c[0].sum()*dx**2/L**2)

W = 2.0 #Constant related to bulk free energy
M = 1.0 # mobility is kept constant, we will vary it later
kappa = 0.5 #gradient coeficient

kx = ky = np.fft.fftfreq(N, d=dx)*2*np.pi   #fourier space frequencies
K = np.array(np.meshgrid(kx , ky ,indexing ='ij'), dtype=np.float32)    #2D grid of Fourier
K2 = np.sum(K*K,axis=0, dtype=np.float32)

# The anti-aliasing factor  
kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
dealias = np.array((np.abs(K[0]) < kmax_dealias )*(np.abs(K[1]) < kmax_dealias ),dtype =bool)

"""
 The interfacial free energy density f(c) = Wc^2(1-c)^2
"""
def finterf(c_hat):
    return kappa*ifft2(K2*c_hat**2).real 

"""
 The bulk free energy density f(c) = Wc^2(1-c)^2
"""
def fbulk(c):
    return W*c**2*(1-c)*c**2

"""
 The derivative of bulk free energy density f(c) = Wc^2(1-c)^2
"""
def dfdc(c):
    return 2*W*(c*(1-c)**2-(1-c)*c**2)

#For variable mobility
def W_func(c):
    return c**3 * (10 - 15*c + 6*c**2)

def M(c, psi):
    return 1 + psi * W_func(c)


psi = 9.0  # or any other value you want to test

c_hat[:] = fft2(c[0])
for i in tqdm(range(1,Nsteps)):
    mobility = M(c[i - 1], psi)
    dfdc_hat[:] = fft2(dfdc(c[i-1])) # the FT of the derivative
    dfdc_hat *= dealias # dealising
    c_hat[:] = (c_hat-dt*K2*mobility*dfdc_hat)/(1+dt*M*kappa*K2**2) # updating in time
    c[i] = ifft2(c_hat).real # inverse fourier transform
    
print('c = ',c[-1].sum()*dx**2/L**2)

print('relative_error = ',np.abs(c[-1].sum()-c[0].sum())/c[0].sum())

plt.imshow(c[-1],cmap='RdBu_r', vmin=0.0, vmax=1.0)
plt.title('$c_0=%.1f$'% c0)
plt.savefig('cahn-hilliard-c0-%.1f.png'% c0)
plt.show()

from matplotlib import animation
from matplotlib.animation import PillowWriter

# generate the GIF animation

fig, ax = plt.subplots(1,1,figsize=(4,4))
im = ax.imshow(c[0],cmap='RdBu_r', vmin=0.0, vmax=1.0)
cb = fig.colorbar(im,ax=ax, label=r'$c(x,y)$', shrink=0.8)
tx = ax.text(190,20,'t={:.1f}'.format(0.0),
         bbox=dict(boxstyle="round",ec='white',fc='white'))
ax.set_title(r'$c_0=%.1f$'% c0)

def animate(i):
    im.set_data(c[5*i])
    im.set_clim(0.0, 1.0)
    tx.set_text('t={:.1f}'.format(5*i*dt))
    return fig,

ani = animation.FuncAnimation(fig, animate, frames= 199,
                               interval = 50)
ani.save('ch-c0='+str(c0)+'.gif',writer='pillow',fps=24,dpi=100)