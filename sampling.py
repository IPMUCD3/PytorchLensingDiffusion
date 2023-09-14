import os,sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim import Adam

sys.path.append('/home/jarmijo/Pytorch_Diffusion/')

from load_data import *
from SimpleUnet import SimpleUnet
from spectral import make_power_map

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = "/gpfs02/work/jarmijo/KappaMaps/norm_quad_SLICS_Cov/imgs/"
#dataset = "/gpfs02/work/jarmijo/KappaMaps/SLICS_Cov/"
weights_dir =  '/home/jarmijo/Pytorch_Diffusion/weights/'
batch_size = 10
gaussian_prior = False
gaussian_path = "/gpfs02/work/jarmijo/data/jaxPS_SLICScovLR_mean_5deg_360x360.npy"
#gaussian_path = "/gpfs02/work/jarmijo/data/jaxPS_SLICScovLR_mean_5deg.npy"
map_size = 360
resolution = 0.29297
N_imgs = 10 

pixel_size = torch.pi * resolution / 180. / 60. #rad/pixel

if gaussian_prior:
    print('Adding power spectrum information...')
    ps_data = np.load(gaussian_path).astype('float32')
    ell = np.array(ps_data[:,0])
    ps_halofit = np.array(ps_data[:,1] / pixel_size**2) # normalisation by pixel size
    # convert to pixel units of our simple power spectrum calculator
    kell = ell /2/np.pi * 360 * pixel_size / map_size
    # Interpolate the Power Spectrum in Fourier Space
    power_map = torch.Tensor(make_power_map(ps_halofit,map_size, kps=kell)).reshape(1,360,360)
    power_map = power_map.to(device)

def log_gaussian_prior(map_data, sigma):
    data_ft = torch.fft.fft2(map_data) / float(map_size)
    ps_fft = torch.real(data_ft*torch.conj(data_ft))
    output = -0.5*torch.sum(ps_fft / (power_map+sigma**2))
    return output
gaussian_prior_score = torch.vmap(torch.func.grad(func=log_gaussian_prior))

def score_fn(y,s,t):
    if gaussian_prior:
        gaussian_score = gaussian_prior_score(y, s)
        net_input = torch.cat([y, s**2*gaussian_score],axis=1).to(device)

        noise = model(net_input,t)
    else:
        noise = model(y,t)
        gaussian_score = torch.zeros_like(y)
    return noise, gaussian_score

print("Loading data...")
train_set = load_diffused_data(name=dataset,N=N_imgs,crop_width=map_size)
trainset_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=True,)

batch = next(iter(trainset_loader))

print("Loading network...")
model = SimpleUnet(ch_inp=1)
model.load_state_dict(torch.load(weights_dir+'DM-gs_train_Nimgs10000.pt'))
model.eval()
model = model.to(device)

x = batch['x']
x0 = x[0]
u0 = batch['u'][0]
x_schedule = []
for i in range(T):
    x_schedule.append(sqrt_alphas_cumprod[i]*x0 + sqrt_one_minus_alphas_cumprod[i]*u0)

tbins=10
T_s = torch.linspace(0,T-1,tbins,dtype=int)#Create time steps from the schedule
i = 1
y = x_schedule[T_s[i]][0].reshape(1,1,360,360)
s = betas[T_s[i]].reshape(1,1,1)
t = T_s[i].reshape(1,)

y = y.to(device)
s = s.to(device)
t = t.to(device)
eps,gs = score_fn(y,s,t)
eps = eps[0].cpu().detach()


step = T_s[2]
x_rec = []
while step > 0:
    sqrt_alpha_inv = 1/torch.sqrt(alphas[step])
    alpha_coef = (1 - alphas[step])/np.sqrt(1 - alphas_cumprod[step])
    x_t_1 = sqrt_alpha_inv*(x_schedule[step] - alpha_coef*eps)
    x_rec.append(x_t_1)
    step -= 1

sqrt_alpha_inv = 1/torch.sqrt(alphas[step])
alpha_coef = (1 - alphas[step])/np.sqrt(1 - alphas_cumprod[step])

x_t_1 = sqrt_alpha_inv*(x_schedule[step] - alpha_coef*eps)

x0_n = x0.numpy()
x_s = x_schedule[step].numpy()
eps_0 = eps.numpy()
u0_n = u0.numpy()


f,ax = plt.subplots(1,3,figsize=(12,4))
ax[0].imshow(x0[0],vmin=-3,vmax=3)
ax[1].imshow(x_schedule[step][0],vmin=-3,vmax=3)
ax[2].hist(x0[0].flatten(),range=(-5,5),bins=35,histtype='step',lw=2,color='k')
ax[2].hist(x_schedule[step][0].flatten(),range=(-5,5),bins=35,histtype='step',lw=2,color='r')
ax[2].set_yscale('log')
#ax.set_ylim(1e1,1.6e4)
plt.tight_layout()
plt.show()


f,ax = plt.subplots(1,3,figsize=(12,4))
eps = eps.cpu()
ax[0].imshow(x_schedule[step-1][0],vmin=-3,vmax=3)
ax[1].imshow(x_t_1[0],vmin=-3,vmax=3)
ax[2].hist(x_schedule[step-1][0].flatten(),range=(-5,5),bins=35,histtype='step',lw=2,color='r')
ax[2].hist(x_t_1[0].flatten(),range=(-5,5),bins=35,histtype='step',lw=2,color='b')
#ax.set_yscale('log')
#ax.set_ylim(1e1,1.6e4)
plt.tight_layout()
plt.show()