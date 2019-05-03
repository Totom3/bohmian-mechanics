import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import os
import random

torch.set_default_dtype(torch.float64)

xi = -20. #left endpont
xf = 20. #right endpoint
J = 5001 #number of space nodes
Xgrid = np.linspace(xi, xf, J)

x = np.load('tracks/try0.npy')

nt = x.shape[1]
N = x.shape[2]

batch_size = 60
# make a random batch

def batch(x):
    batchy = np.zeros([batch_size, 2, nt, 3])
    true_next = np.zeros([batch_size, nt, 1])
    for i in range(batch_size):
        k = np.random.randint(0, N-3)
        batchy[i, :, :, :] = x[:, :, k:k+3]
        true_next[i, :, 0] = x[0, :, k+3]
    return batchy, true_next

class TNet(nn.Module):

    def __init__(self):
        super(TNet, self).__init__()
        self.pad1 = nn.ReplicationPad2d((0, 0, 1, 1))
        self.pad2 = nn.ReflectionPad2d((0, 0, 1, 1))
        self.pad3 = nn.ReplicationPad1d(1)
        self.pad4 = nn.ReflectionPad1d(1)
        self.con1 = nn.Conv2d(2, 9, kernel_size = (3,3))
        self.con2 = nn.Conv1d(9, 18, kernel_size = 3)
        self.con3 = nn.Conv1d(18, 27, kernel_size = 3)
        self.con4 = nn.Conv1d(27, 36, kernel_size = 3)
        self.nonlin = nn.SELU()
        self.drop = nn.Dropout(0.03)
        self.fct = nn.Linear(36, 1)


    def forward(self, x):
        x = 2*self.pad1(x) - self.pad2(x)
        x = self.con1(x)
        x = torch.squeeze(x)
        x = 2*self.pad3(x) - self.pad4(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.con2(x)
        x = 2*self.pad3(x) - self.pad4(x)
        x = self.nonlin(x)
        x - self.drop(x)
        x = self.con3(x)
        x = 2*self.pad3(x) - self.pad4(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.con4(x)
        x = self.nonlin(x)
        x - self.drop(x)
        x = x.transpose(1,2)
        x = self.fct(x)
        return x


net = TNet()
net = net.cuda()
net.load_state_dict(torch.load('saves/tnet'))
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00002)

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

directory = "tracks"

asps = []

# for i in range(1000):
for file in files("./{}/".format(directory)):
    asps.append(file)

for i in range(10000):
    ranfi = random.choice(asps)
    print("file name:" + ranfi)
    x = np.load(("./{}/".format(directory) + ranfi))
    for epoch in range(0, 10_000+1):
        optimizer.zero_grad()
    
        data, true_next = batch(x)
        data = torch.from_numpy(data).cuda()
        true_next = torch.from_numpy(true_next).cuda()
        
        output = net(data)

        loss = criterion(output, true_next)
        loss.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print("Epoch "+ str(epoch) + ": "+("%.9f" % loss.data.item()))

    
        if epoch % 5000 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 1.001
            torch.save(net.state_dict(), "saves/tnet")

