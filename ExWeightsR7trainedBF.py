#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:26:06 2020

@author: leoniddatta
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transform 
from torchvision import datasets, transforms
import tensorflow as tf
import torch.utils.data as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math


class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        st=2
        self.conv1 = nn.Conv2d(3, 4, 3, 1 )
        self.conv2 = nn.Conv2d(4, 8, 3, st )
        self.conv3 = nn.Conv2d(8, 16, 3, st )
        self.conv4 = nn.Conv2d(16, 32, 3, st )
        self.conv5 = nn.Conv2d(32, 10, 3, st )
        self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
        
    def forward(self, x):
        
        x=x.float()
        x=self.conv1(x) 
        x = F.relu(x) 
        x=self.conv2(x) 
        x = F.relu(x)
        x=self.conv3(x) 
        x = F.relu(x)
        x=self.conv4(x) 
        x = F.relu(x)
        x=self.conv5(x) 
        x = F.relu(x)
        x = self.GAP(x)
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x   
    
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

bb=datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])).test_data.numpy()

b2=datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])).test_labels.numpy()

cc=datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])).test_data.numpy()

c2=datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])).test_labels.numpy()


b=np.zeros((10000,56,56))
c=np.zeros((10000,56,56))

for i in range (bb.shape[0]):
    b[i]=cv2.resize(bb[i], (56, 56))
    

for i in range (cc.shape[0]):
    c[i]=cv2.resize(cc[i], (56, 56))

b=b/255
c=c/255
datasize=56

b=1*(b>0.3)
c=1*(c>0.3)

mask=np.zeros((datasize,datasize))

maskgap=5

for i in range(0,datasize,maskgap):
    for j in range(0,datasize,maskgap):
        mask[i,j]=1
        #if(i!=datasize or j!=datasize):
             
mask[:,0]=0
mask[0,:]=0
mask[:,55]=0
mask[55,:]=0
plt.imshow(mask, cmap='gray',  interpolation='nearest')
plt.show()


b=b*mask
c=c*mask

                


for k in range(b.shape[0]):
    for i in range(0,datasize,maskgap):
        for j in range(0,datasize,maskgap):
            if(b[k,i,j]==1):
                b[k,i,j]=0
                b[k,i,j-1]=1
                b[k,i-1,j-1]=1
                b[k,i+1,j-1]=1
                b[k,i+1,j]=1
                
for k in range(c.shape[0]):
    for i in range(0,datasize,maskgap):
        for j in range(0,datasize,maskgap):
            if(c[k,i,j]==1):
                c[k,i,j]=0
                c[k,i-1,j]=1
                c[k,i-1,j+1]=1
                c[k,i,j+1]=1
                c[k,i+1,j+1]=1
                


bbb=b
ccc=c
inchan=3
a=np.zeros((60000,56,56,inchan))
b=np.zeros((10000,56,56,inchan))
c=np.zeros((10000,56,56,inchan))

b[:,:,:,1]=bbb
c[:,:,:,0]=ccc


fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
fig.suptitle('Test data1 - green curve accuracy in graph')
fig.set_figheight(4)
fig.set_figwidth(6)
ax1.imshow(b[0], cmap='gray',  interpolation='nearest')
ax2.imshow(b[1], cmap='gray',  interpolation='nearest')
ax3.imshow(b[2], cmap='gray',  interpolation='nearest')
ax4.imshow(b[3], cmap='gray',  interpolation='nearest')
ax5.imshow(b[4], cmap='gray',  interpolation='nearest')
ax6.imshow(b[5], cmap='gray',  interpolation='nearest')
plt.show()

fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
fig.suptitle('Test data2 - red curve accuracy in graph')
fig.set_figheight(4)
fig.set_figwidth(6)
ax1.imshow(c[0], cmap='gray',  interpolation='nearest')
ax2.imshow(c[1], cmap='gray',  interpolation='nearest')
ax3.imshow(c[2], cmap='gray',  interpolation='nearest')
ax4.imshow(c[3], cmap='gray',  interpolation='nearest')
ax5.imshow(c[4], cmap='gray',  interpolation='nearest')
ax6.imshow(c[5], cmap='gray',  interpolation='nearest')
plt.show()

  
b=np.transpose(b, (0,3, 1, 2))
c=np.transpose(c, (0,3, 1, 2))
       
data2=torch.from_numpy(b)
target2=torch.from_numpy(b2)
target2 = torch.tensor(target2, dtype=torch.long)
my_testdataset = utils.TensorDataset(data2,target2)
data2=torch.from_numpy(c)
target2=torch.from_numpy(c2)
target2 = torch.tensor(target2, dtype=torch.long)
my_hortestdataset = utils.TensorDataset(data2,target2)

grntest_loader = torch.utils.data.DataLoader(my_testdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
redtest_loader = torch.utils.data.DataLoader(my_hortestdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)


def test(args, model, device, test_loader):
    model.train(mode=False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc=int(100. * correct / len(test_loader.dataset))
    return acc





in_c2=4
in_c3=8
in_c4=16
in_c5=32
seedItr=1
GLacc=np.zeros((seedItr,in_c2,in_c2,in_c3,in_c3,in_c4,in_c4,in_c5,in_c5))
R7acc=np.zeros((seedItr,in_c2,in_c2,in_c3,in_c3,in_c4,in_c4,in_c5,in_c5))

for seed_no in range(seedItr):
    print("New Seed No is",seed_no+1)
    for i2 in range (in_c2):
        for j2 in range(in_c2):
            model = Netconv()
            model.load_state_dict(torch.load('R7models/BaseR7seed{seed}.pt'.format(seed=seed_no+1)))
            model.conv1.weight[:,0,:,:]=model.conv1.weight[:,0,:,:]+model.conv1.weight[:,1,:,:]
            model.conv1.weight[:,1,:,:]=model.conv1.weight[:,0,:,:]-model.conv1.weight[:,1,:,:]
            model.conv1.weight[:,0,:,:]=model.conv1.weight[:,0,:,:]-model.conv1.weight[:,1,:,:]
            
            if(i2>j2):
                model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]+model.conv2.weight[:,j2,:,:]
                model.conv2.weight[:,j2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                
                for i3 in range (in_c3):
                    for j3 in range(in_c3):
                        if(i3>j3):
                            model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]+model.conv3.weight[:,j3,:,:]
                            model.conv3.weight[:,j3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                            model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                            
                            for i4 in range (in_c4):
                                for j4 in range(in_c4):
                                    if(i4>j4):
                                        model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]+model.conv4.weight[:,j4,:,:]
                                        model.conv4.weight[:,j4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                                        model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                                            
                                        for i5 in range (in_c5):
                                            for j5 in range(in_c5):
                                                if(i5>j5):
                                                    model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]+model.conv5.weight[:,j5,:,:]
                                                    model.conv5.weight[:,j5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                                                    model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                                                    model.cuda()
                                                    print("Seed=",seed_no+1,"conv2 ex",i2,j2,"conv3 ex",i3,j3,"conv4 ex",i4,j4,"conv5 ex",i5,j5)
#                                                     print("R7 Data")
#                                                     R7acc[seed_no,i2,j2,i3,j3,i4,j4,i5,j5]=test(args, model, device, redtest_loader)
                                                    print("GL Data")
                                                    GLacc[seed_no,i2,j2,i3,j3,i4,j4,i5,j5]=test(args, model, device, grntest_loader)
                                                      
                            
                            
            
#np.save('R7trnBFExResR7Seed1.npy',R7acc)
np.save('R7trnBFExResGLSeed1.npy',GLacc)   
    


