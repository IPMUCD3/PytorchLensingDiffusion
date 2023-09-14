import os,sys
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim import Adam

#TODO add to __init__.py
sys.path.append('/home/jarmijo/Pytorch_Diffusion/')

from load_data import *
from SimpleUnet import SimpleUnet
from spectral import make_power_map

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = "/gpfs02/work/jarmijo/KappaMaps/norm_quad_SLICS_Cov/imgs/"
#dataset = "/gpfs02/work/jarmijo/KappaMaps/SLICS_Cov/"
output_dir =  '/home/jarmijo/Pytorch_Diffusion/weights/'
batch_size = 32
gaussian_prior = True
gaussian_path = "/gpfs02/work/jarmijo/data/jaxPS_SLICScovLR_mean_5deg_360x360.npy"
learning_rate = 1e-5
#spectral_norm = 1 Still need to apply!
map_size = 360
resolution = 0.29297 #arcmin/pix
N_imgs = 12000 #num of augmented imgs

pixel_size = torch.pi * resolution / 180. / 60. #rad/pixel

if gaussian_prior:
    print('Adding power spectrum information...')
    ps_data = np.load(gaussian_path).astype('float32')
    ell = np.array(ps_data[:,0])
    # massivenu: channel 4
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

def get_loss(model,y,u,s,t):
    noise_pred,gs = score_fn(y,s,t)
    return F.l1_loss(u,(noise_pred+s*gs))

model = SimpleUnet()
model.to(device)
#print("Num params: ", sum(p.numel() for p in model.parameters()))
optimizer = Adam(model.parameters(), lr=learning_rate)
epochs = 101 # Try more!

print("Loading data...")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop(size=crop_width),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()])

train_set = load_diffused_data(name=dataset,N=N_imgs,crop_width=map_size)
trainset_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=True,)

for epoch in range(epochs):
    for step,batch in enumerate(trainset_loader):
        y = batch['y'].to(device)
        u = batch['u'].to(device)
        s = batch['s'].to(device)
        t = batch['t'].to(device)
        optimizer.zero_grad()
        loss = get_loss(model,y,u,s,t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()} ")
            np.savetxt(output_dir+'current_ep_lo.txt',np.array([epoch,loss.item()]))
#        sample_plot_image()
    if epoch % 50 == 0:
        torch.save(model.state_dict(), output_dir+'DM+gs_train_Nimgs10000_%d.pt'%epoch)

if gaussian_prior:
    torch.save(model.state_dict(), output_dir+'DM+gs_train_Nimgs10000.pt')
else:
    torch.save(model.state_dict(), output_dir+'DM-gs_train_Nimgs10000.pt')

