import os,sys
import numpy as np
import torch

class load_data():
    def __init__(self,name):
        self.data_dir = name
        self.list_data = os.listdir(name)
        self.dataset = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
        self.x = None
        
    def sample(self,transform,shape,N, crop_width):
        self.N = N
        self.map_size = crop_width
        map_ids = torch.randint(low=0,high=len(self.list_data),size=(self.N,))
        self.map_ids = map_ids
        data_N = np.array(self.dataset)[self.map_ids.numpy()]
        ds = [transform(s) for s in data_N]
        ds = torch.cat(ds)
        ds = ds.reshape(shape)
        self.x = ds

    def __getitem__(self,index):
        x_i = self.x[index]
        return x_i
    
    def __len__(self):
        return len(self.x)
        
class load_diffused_data():
    def __init__(self,name,N, crop_width):
        super().__init__(name,N, crop_width)
        
    def schedule(T,noise_dist_std):  
        self.T = T
        self.noise_dist_std = noise_dist_std

        self.betas = torch.linspace(0, noise_dist_std, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def sample(self,transform,shape):
        map_ids = torch.randint(low=0,high=len(self.list_data),size=(self.N,))
        self.map_ids = map_ids
        data_N = np.array(self.dataset)[self.map_ids.numpy()]        
        ds = [transform(s) for s in data_N]
        ds = torch.cat(ds)
        ds = ds.reshape(shape)
    
        u = torch.randn((shape))
        t = torch.randint(low=0,high=self.T,size=(len(ds),))

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape((len(ds),1, 1,1))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape((len(ds),1, 1,1))
        beta_t = self.posterior_variance[t].reshape((len(ds),1, 1,1))

        y = sqrt_alphas_cumprod_t * ds + sqrt_one_minus_alphas_cumprod_t * u
        
        self.x = ds
        self.y = y
        self.u = u
        self.t = t
        self.beta_t = beta_t
        
    def __getitem__(self,index):
        x_i = self.x[index]
        y_i = self.y[index]
        u_i = self.u[index]
        t_i = self.t[index]
        beta_t_i = self.beta_t[index]
        
        return {'x':x_i, 'y':y_i, 'u':u_i,'t':t_i,'s':beta_t_i}
    
    def __len__(self):
        return len(self.x)