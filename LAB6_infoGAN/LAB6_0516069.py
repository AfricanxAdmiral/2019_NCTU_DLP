from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda")

## Hyper Parameters
Dataset = 'mnist'
dataroot = './data'
workers = 2
batchSize = 100
noiseSize = 10 ## total 10*10
imageSize = 64
nz = 64
nc = 1
ngf = 64
ndf = 64
niter = 80
lr_d = 2e-4
lr_q_g =1e-3
beta1 = 0.5
out_model = './model'
out_image = './tmp'
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataset = dset.MNIST(root=dataroot, download=False,
                     transform=transforms.Compose([
                               transforms.Resize(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                            ])
                    )

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batchSize,
                                         shuffle=True,
                                         num_workers=int(workers))
										 
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

## Generate Noise Sample with Constrain
def noise_sample(dis_c, noise, batch_size):

    idx = np.random.randint(10, size=batch_size)
    c = np.zeros((batch_size, 10))
    c[range(batch_size),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    noise.data.copy_(torch.Tensor(np.random.randn(batch_size, nz-10)))
    z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)

    return z, idx

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.D = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(
            nn.Linear(8192, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        output = self.main(input)
        out_D = self.D(output).view(-1, 1)
        out_Q = self.Q(output.view(output.size(0), -1)).squeeze()
        return out_D, out_Q
		
net_G = Generator().to(device)
net_G.apply(weights_init)
net_D = Discriminator.to(device)
net_D.apply(weights_init)

criterion_D = nn.BCELoss().cuda()
criterion_Q_dis = nn.CrossEntropyLoss().cuda()

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam([{'params':net_D.main.parameters()}, {'params':net_D.D.parameters()}], lr=lr_d, betas=(beta1, 0.999))
optimizerG = optim.Adam([{'params':net_G.parameters()}, {'params':net_D.Q.parameters()}], lr=lr_q_g, betas=(beta1, 0.999))

# Reusable space in GPU
real_x = torch.FloatTensor(batchSize, 1, imageSize, imageSize).cuda()
label = torch.FloatTensor(noiseSize**2, 1).cuda()
dis_c = torch.FloatTensor(noiseSize**2, 10).cuda()
noise = torch.FloatTensor(noiseSize**2, batchSize-10).cuda()

real_x = Variable(real_x)
label = Variable(label, requires_grad=False)
dis_c = Variable(dis_c)
noise = Variable(noise)

# fixed random variables
idx = np.arange(noiseSize).repeat(noiseSize).reshape(noiseSize, noiseSize).transpose().reshape(noiseSize**2)
one_hot = np.zeros((noiseSize**2, 10))
one_hot[range(noiseSize**2), idx] = 1
fix_noise = torch.Tensor(np.random.normal(0, 1, (noiseSize, nz-10)).repeat(noiseSize, axis=0))

D_losses = []
G_losses = []
Q_losses = []

p_real = []
p_fake_before = []
p_fake_after = []

correct_real = 0
total_real = 0
correct_fake_before = 0
total_fake_before = 0
correct_fake_after = 0
total_fake_after = 0

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        optimizerD.zero_grad()
        
        real_dig = data[0].to(device)
        
        batch_size = real_dig.size(0)
        
        label.data.resize_(batch_size, 1)
        dis_c.data.resize_(batch_size, 10)
        noise.data.resize_(batch_size, 54)

        ## Real
        #real_out = net_FE(real_dig)
        #output_t = net_D(real_out)
        output_t, _ = net_D(real_dig)
        label.data.fill_(real_label)
        loss_real = criterion_D(output_t, label)
        loss_real.backward()
        # D_x = output.mean().item()
        
        total_real += batch_size
        for j in range(batch_size):
            if output_t[j]>=0.5:
                correct_real += 1
        if i %100 == 0:
            p_real.append(100.*correct_real/total_real)
            correct_real = 0
            total_real = 0

        ## Fake
        z, idx = noise_sample(dis_c, noise, batch_size)
        fake_dig = net_G(z)
        label.data.fill_(fake_label)
        #fake_out = net_FE(fake_dig.detach())
        #output_f = net_D(fake_out)
        output_f, _ = net_D(fake_dig.detach())
        loss_fake = criterion_D(output_f, label)
        loss_fake.backward()
        # D_G_z1 = output.mean().item()
        
        
        total_fake_before += batch_size
        for j in range(batch_size):
            if output_f[j]>=0.5:
                    correct_fake_before += 1
        if i %100 == 0:
            p_fake_before.append(100.*correct_fake_before/total_fake_before)
            correct_fake_before = 0
            total_fake_before = 0
            
        
        ## Maximize log(D(x)) + log(1 - D(G(z)))
        loss_D = loss_real + loss_fake
        if i % 100 == 0:
            D_losses.append(loss_D)
        
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        fake_out = net_FE(fake_dig)
        #output_r = net_D(fake_out)
        #output_q = net_Q(fake_out)
        output_r, output_q = net_D(fake_dig.detach())
        loss_reconstruck = criterion_D(output_r, label)
        if i % 100 == 0:
            G_losses.append(loss_reconstruck)
        
        total_fake_after += batch_size
        for j in range(batch_size):
            if output_r[j]>=0.5:
                    correct_fake_after += 1
        if i %100 == 0:
            p_fake_after.append(100.*correct_fake_after/total_fake_after)
            correct_fake_after = 0
            total_fake_after = 0
        
        # D_G_z2 = output.mean().item()
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        loss_q = criterion_Q_dis(output_q, target)
        if i % 100 == 0:
            Q_losses.append(loss_q)
        
        loss_G = loss_reconstruck + loss_q
        loss_G.backward()
        
        optimizerG.step()

        if i % 100 == 99: 
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'%(epoch+1, niter, i+1, len(dataloader), loss_D.item(), loss_G.item()))
        
    if epoch % 3 == 2:
        ## Load fixed z
        noise.data.resize_(noiseSize**2, 54)
        dis_c.data.resize_(noiseSize**2, 10)
        noise.data.copy_(fix_noise)
        dis_c.data.copy_(torch.Tensor(one_hot))
        z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
            
        gen_image = net_G(z)
        save_image(gen_image.detach(), out_image+'/c1_%03d.png' % (epoch+1), normalize=True, nrow=noiseSize)

        # do checkpointing
        torch.save(net_G.state_dict(), '%s/net_epoch_%03d.pkl' % (out_model, epoch))