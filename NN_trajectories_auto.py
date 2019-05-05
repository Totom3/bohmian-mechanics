import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import os
import random

torch.set_default_dtype(torch.float64)

nt = 500
N = 1000

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file
            
directory = "tracks"

asps = []

for file in files("./{}/".format(directory)):
    asps.append(file)

length = len(asps)

z = np.zeros([length, 2, nt, N])
for i in range(length):
    y = np.load("./{}/{}".format(directory, asps[i]))
    z[i, :, :, :] = y 
    
batch_size = 200
# make a random batch

def batch(z):
    batchy = np.zeros([batch_size, 2, nt, 9])
    df = np.zeros([batch_size, nt])
    for i in range(batch_size):
        j = np.random.randint(length)
        k = np.random.randint(0, N-9)
        batchy[i, :, :, :] = z[j, :, :, k:k+9]
        df[i, :] = z[j, 0, :, k+9] - z[j, 0, :, k+8]
    return batchy, df

class TNet(nn.Module):

    def __init__(self):
        super(TNet, self).__init__()
        self.con1 = nn.Conv2d(2, 18, kernel_size = (1,5))
        self.con2 = nn.Conv2d(18, 36, kernel_size = (1,5))
        self.nonlin = nn.SELU()
        self.drop = nn.Dropout(0.03)
        self.fct = nn.Linear(nt, nt)
        self.final = nn.Linear(36, 1)


    def forward(self, x):
        x = self.con1(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.con2(x)
        x = torch.squeeze(x, 3)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.fct(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = x.transpose(1,2)
        x = self.final(x)
        x = torch.squeeze(x, 2)
        return x


net = TNet()
net = net.cuda()
net.load_state_dict(torch.load('saves/tnet'))
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


for epoch in range(0, 900_000+1):

    optimizer.zero_grad()
    
    data, df = batch(z)
    data = torch.from_numpy(data).cuda()
    df = torch.from_numpy(df).cuda()
        
    output = net(data)

    loss = 1000000*criterion(output, df)
    loss.backward()
    optimizer.step()
        
    if epoch % 100 == 0:
        print("Epoch "+ str(epoch) + ": "+("%.9f" % loss.data.item()))

    
    if epoch % 100 == 0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 1.003
        print("learning rate {:4f}".format(g['lr']))
        torch.save(net.state_dict(), "saves/tnet")

