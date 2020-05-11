
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:44:53 2020
@author: leoniddatta
"""

#from __future__ import print_function
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
import scipy.ndimage.filters as C
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import copy
import math

GL=0 #SET GL=0 for Red-7-shaped training Data , Set GL=1 for Green-L-shaped training Data

#Code for validating the base model
def weightittensor(inc,outc,k,g): #Function for weight initialization. inc=input_channel, outc=output_channel, k=kernel size, g=group
    weightrange=1. / math.sqrt(inc*k*k)
    if(inc==g):
        inc=1
    kernel=torch.FloatTensor(outc,inc,k, k).uniform_(-weightrange, weightrange)
    return kernel
def biasittensor(inc,outc,k,g): #Function for weight initialization. inc=input_channel, outc=output_channel, k=kernel size, g=group
    weightrange=inc*k*k
    weightrange= 1. / math.sqrt(weightrange)
    
    kernel=torch.FloatTensor(outc).uniform_(-weightrange, weightrange)
    return kernel
def weightittensornotfull(inc,outc,k,g): #Function for weight initialization. inc=input_channel, outc=output_channel, k=kernel size, g=group
    weightrange=1. / math.sqrt(inc*k*k)
    if(inc==g):
        inc=1
    kernel=torch.FloatTensor(outc,k, k).uniform_(-weightrange, weightrange)
    return kernel


def add_channel(x,n):
    #x=x.detach().numpy() 
# =============================================================================
#     n1=int(x.shape[1]/n) #n is no of output channels
#     y=torch.zeros((x.shape[0],n,x.shape[2],x.shape[3]))
#     for i in range(n):
#         for j in range(n1):
#             y[:,i,:,:]=y[:,i,:,:]+x[:,(i*n1)+j,:,:]
#     #y=torch.from_numpy(y)
# =============================================================================
    ran=int(x.shape[1]/2)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(j<(ran)):
                #print(j)
                x[i,j,:,:]=x[i,j*2,:,:]+x[i,((j*2)+1),:,:]
    
    return x[:,0:ran,:,:]

def rearrange(x,n):
    z=x
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    z=torch.flatten(z, start_dim=2)
    for i in range(z.shape[0]):
        sim=torch.zeros((z.shape[1],z.shape[1]))

        for j in range(z.shape[1]):
            for k in range(z.shape[1]):
                if((j==k) or (j<k)):
                    sim[j,k]=-1.0000
                else:
                    sim[j,k]=cos(z[i,j],z[i,k])
        #print("initial similarity matrix",sim)
        max=torch.ones(z.shape[1])*sim[0,0]
        maxindj=torch.zeros(z.shape[1])
        maxindk=torch.zeros(z.shape[1])
        ind=int(z.shape[1]/2)
        for l in range(ind):
            for j in range(z.shape[1]):
                for k in range(z.shape[1]):
                    if (sim[j,k]>max[l]):
                        max[l]=sim[j,k]
                        maxindj[l]=j
                        maxindk[l]=k
            maxindj=maxindj.long() 
            maxindk=maxindk.long()
            #print(maxindj)
            #print(maxindk)  
            sim[:,maxindj[l]]=-2.0000
            sim[maxindk[l],:]=-2.0000       
            sim[maxindj[l],:]=-2.0000
            sim[:,maxindk[l]]=-2.0000
            #print(sim)
            #print("max",max)
        #print(maxindj)
        #print(maxindk)
        
        sortedindex=torch.zeros((z.shape[1])).long()
        
        for l in range(ind):
            sortedindex[l*2]=maxindj[l]
            sortedindex[(l*2)+1]=maxindk[l]
        for l in range(z.shape[1]):
            x[i,l]=x[i,sortedindex[l]]
        #print(sortedindex)
       
      
    
#Function for plotting the graph
def plotgraph (xs,y1s,y2s,yts):
    plt.clf()
    fig = plt.figure(figsize=(10, 7))
    plt.plot(xs,y1s,'s:r')
    plt.ylim(0, 100)
    #plt.xlim(0,20)
    fig.suptitle('ACCURACY GRAPH')
    plt.xlabel('Epoch no')
    plt.ylabel('Accuracy %')
    for x,y in zip(xs,y1s):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

    plt.plot(xs,y2s,'^:g')
    for x,y in zip(xs,y2s):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    plt.plot(xs,yts,'*:b')
    for x,y in zip(xs,yts):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
    blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=10, label='training data')
    red_line = mlines.Line2D([], [], color='red', marker='s',
                          markersize=10, label='R7 test data')
    green_line = mlines.Line2D([], [], color='green', marker='^',
                          markersize=10, label='GL test data')
    plt.legend(handles=[blue_line,red_line,green_line],loc=2)
    plt.show()
k=4

#Line 91-123 for the model using nn.conv2d
# =============================================================================
# class Netconv(nn.Module):
#     def __init__(self):
#         super(Netconv, self).__init__()
#         st=2
#         self.conv1 = nn.Conv2d(3, 16*k, 3, 1). #K value is set one for base model. 
#         self.conv2 = nn.Conv2d(16*k, 32*k, 3, st)
#         self.conv3 = nn.Conv2d(32*k, 64*k, 3, st)
#         self.conv4 = nn.Conv2d(64*k, 128*k, 3, st)
#         self.conv5 = nn.Conv2d(128*k, 10, 3, st)
#         self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
#     def forward(self, x):
#         x=x.float()
#         x=self.conv1(x) 
#         x = F.relu(x)
#         x=self.conv2(x) 
#         x = F.relu(x)
#         x=self.conv3(x) 
#         x = F.relu(x)
#         x=self.conv4(x) 
#         x = F.relu(x)
#         x=self.conv5(x) 
#         x = F.relu(x)
#         x = self.GAP(x)
#         x = x.view(-1, 10) 
#         x=F.log_softmax(x, dim=1)
#         return x 
# =============================================================================
#Line 120 to 160 for model using torch.nn.functional.conv2d
add=1
class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        self.x1=torch.nn.Parameter(weightittensornotfull(3,16*k,3,1)) # random tensor weights generated for layer 1
        self.x2=torch.nn.Parameter(weightittensornotfull(int(16*k/add),32*k,3,1)) #random tensor weights generated for layer 2
        self.x3=torch.nn.Parameter(weightittensornotfull(int(32*k/add),64*k,3,1)) #random tensor weights generated for layer 3
        self.x4=torch.nn.Parameter(weightittensornotfull(int(64*k/add),128*k,3,1)) # random tensor weights generated for laye 4
        self.x5=torch.nn.Parameter(weightittensornotfull(int(128*k/add),10,3,1)) # random tensor weights generated for layer 5
        self.bias1=torch.nn.Parameter(biasittensor(1,16*k,3,1))
        self.bias2=torch.nn.Parameter(biasittensor(int(16*k/add),32*k,3,1))
        self.bias3=torch.nn.Parameter(biasittensor(int(32*k/add),64*k,3,1))
        self.bias4=torch.nn.Parameter(biasittensor(int(64*k/add),128*k,3,1))
        self.bias5=torch.nn.Parameter(biasittensor(int(128*k/add),10,3,1))
        self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
        
    def forward(self, x):
        new_shape1 = [16*k,3,3,3]
        new_shape2 = [32*k,16*k,3,3]
        new_shape3 = [64*k,32*k,3,3]
        new_shape4 = [128*k,64*k,3,3]
        new_shape5 = [10,128*k,3,3]
        #self.x1=torch.reshape(self.x1,(16,3,3))
        x1_weights = torch.zeros(new_shape1, dtype=torch.float32)
        x2_weights = torch.zeros(new_shape2, dtype=torch.float32)
        x3_weights = torch.zeros(new_shape3, dtype=torch.float32)
        x4_weights = torch.zeros(new_shape4, dtype=torch.float32)
        x5_weights = torch.zeros(new_shape5, dtype=torch.float32)
        
        
        
        for i in range(x1_weights.shape[1]): # amount of output channels
            x1_weights[:,i] = self.x1
        for i in range(x2_weights.shape[1]): # amount of output channels
            x2_weights[:,i] = self.x2
        for i in range(x3_weights.shape[1]): # amount of output channels
            x3_weights[:,i] = self.x3
        for i in range(x4_weights.shape[1]): # amount of output channels
            x4_weights[:,i] = self.x4
        for i in range(x5_weights.shape[1]): # amount of output channels
            x5_weights[:,i] = self.x5
            
            
# =============================================================================
#         for i in range(int(x2_weights.shape[1]/2)): # amount of output channels
#             x2_weights[:,i*2] = self.x2[:,i]
#             x2_weights[:,(i*2)+1] = self.x2[:,i]
#         for i in range(int(x3_weights.shape[1]/2)): # amount of output channels
#             x3_weights[:,i*2] = self.x3[:,i]
#             x3_weights[:,(i*2)+1] = self.x3[:,i]
#         for i in range(int(x4_weights.shape[1]/2)): # amount of output channels
#             #x4_weights[:,i] = self.x4#self.x4
#             x4_weights[:,i*2] = self.x4[:,i]
#             x4_weights[:,(i*2)+1] = self.x4[:,i]
#         for i in range(int(x5_weights.shape[1]/2)): # amount of output channels
#             #x5_weights[:,i] = self.x5#self.x5
#             x5_weights[:,i*2] = self.x5[:,i]
#             x5_weights[:,(i*2)+1] = self.x5[:,i]
# =============================================================================
        
            
        x1_weights = x1_weights.cuda()
        x2_weights = x2_weights.cuda()
        x3_weights = x3_weights.cuda()
        x4_weights = x4_weights.cuda()
        x5_weights = x5_weights.cuda()
        
        x=x.float()
        x=torch.nn.functional.conv2d(x, x1_weights,self.bias1,stride=1)
        x = F.relu(x)
        #rearrange(x,2)
        x=x.float()
        x=torch.nn.functional.conv2d(x, x2_weights,self.bias2,stride=2)
        x = F.relu(x)
        #rearrange(x,2)
        x=x.float()
        x=torch.nn.functional.conv2d(x, x3_weights,self.bias3,stride=2)
        x = F.relu(x)
        #rearrange(x,2)
        x=x.float()
        x=torch.nn.functional.conv2d(x, x4_weights,self.bias4,stride=2)
        x = F.relu(x)
        #rearrange(x,2)
        x=x.float() 
        x=torch.nn.functional.conv2d(x, x5_weights,self.bias5,stride=2)
        x = F.relu(x) 
        x = self.GAP(x)
        x = x.view(-1, 10)
        x=F.log_softmax(x, dim=1)
        return x
    


    def parameters(self):
        return [self.x1, self.x2,self.x3,self.x4,self.x5,self.bias1,self.bias2,self.bias3,self.bias4,self.bias5] 

def train(args, model, device, train_loader, optimizer, epoch, hortest_loader,test_loader):
    r=0
    g=0
    t=0
    running_loss = 0
    total_train = 0
    correct_train = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += predicted.eq(target.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        q=int(100. * batch_idx / len(train_loader))

        if batch_idx % args.log_interval == 0:
            if batch_idx % args.log_interval == 0:
                if (q%10) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \taccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),train_accuracy))
                if (q==50):
                    r=test(args, model, device, hortest_loader)
                    g=test(args, model, device, test_loader)
                    t=int(train_accuracy)
    return [r,g,t,train_accuracy]

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

def main():

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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),


        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    aa=datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])).train_data.numpy()

    a2=datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])).train_labels.numpy()

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

    a=np.zeros((60000,56,56))
    b=np.zeros((10000,56,56))
    c=np.zeros((10000,56,56))
    for i in range (aa.shape[0]):
        a[i]=cv2.resize(aa[i], (56, 56))

    for i in range (bb.shape[0]):
        b[i]=cv2.resize(bb[i], (56, 56))

    for i in range (cc.shape[0]):
        c[i]=cv2.resize(cc[i], (56, 56))
    
    a=a/255
    b=b/255
    c=c/255
    datasize=56

    a=1*(a>0.3)
    b=1*(b>0.3)
    c=1*(c>0.3)

# =============================================================================
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

    a=a*mask
    b=b*mask
    c=c*mask
# =============================================================================


   #L SHAPE

# =============================================================================
    if(GL==1):
        for k in range(a.shape[0]):
            for i in range(0,datasize,maskgap):
                for j in range(0,datasize,maskgap):
                    if(a[k,i,j]==1):
                        a[k,i,j]=0
                        a[k,i,j-1]=1
                        a[k,i-1,j-1]=1
                        a[k,i+1,j-1]=1
                        a[k,i+1,j]=1

    else:
        for k in range(a.shape[0]):
            for i in range(0,datasize,maskgap):
                for j in range(0,datasize,maskgap):
                    if(a[k,i,j]==1):
                        a[k,i,j]=0
                        a[k,i-1,j]=1
                        a[k,i-1,j+1]=1
                        a[k,i,j+1]=1
                        a[k,i+1,j+1]=1



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
# =============================================================================

    aaa=a
    bbb=b
    ccc=c

    a=np.zeros((60000,56,56,3))
    b=np.zeros((10000,56,56,3))
    c=np.zeros((10000,56,56,3))


    if(GL==1):
        a[:,:,:,1]=aaa
    else:
        a[:,:,:,0]=aaa

    b[:,:,:,1]=bbb
    c[:,:,:,0]=ccc

    print("train data is")
    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Training Data (blue curve train accyracy in graph)')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(a[0], cmap='gray',  interpolation='nearest')
    ax2.imshow(a[1], cmap='gray',  interpolation='nearest')
    ax3.imshow(a[2], cmap='gray',  interpolation='nearest')
    ax4.imshow(a[3], cmap='gray',  interpolation='nearest')
    ax5.imshow(a[4], cmap='gray',  interpolation='nearest')
    ax6.imshow(a[5], cmap='gray',  interpolation='nearest')
    plt.show()


    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Test data1 - green curve accuracy in graph')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(b[0], cmap='gray',  interpolation='nearest')
    ax2.imshow(b[1], cmap='gray',  interpolation='nearest')
    ax3.imshow(b[2], cmap='gray',  interpolation='nearest')
    ax4.imshow(b[3], cmap='gray',  interpolation='nearest')
    ax5.imshow(b[4], cmap='gray',  interpolation='nearest')
    ax6.imshow(b[5], cmap='gray',  interpolation='nearest')
    plt.show()

    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Test data2 - red curve accuracy in graph')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(c[0], cmap='gray',  interpolation='nearest')
    ax2.imshow(c[1], cmap='gray',  interpolation='nearest')
    ax3.imshow(c[2], cmap='gray',  interpolation='nearest')
    ax4.imshow(c[3], cmap='gray',  interpolation='nearest')
    ax5.imshow(c[4], cmap='gray',  interpolation='nearest')
    ax6.imshow(c[5], cmap='gray',  interpolation='nearest')
    plt.show()
# =============================================================================
#     a=np.reshape(a,(60000,1,56,56))
#     b=np.reshape(b,(10000,1,56,56))
#     c=np.reshape(c,(10000,1,56,56))
# =============================================================================
    
    a=np.transpose(a, (0,3, 1, 2))
    b=np.transpose(b, (0,3, 1, 2))
    c=np.transpose(c, (0,3, 1, 2))


    data=torch.from_numpy(a)
    target=torch.from_numpy(a2)
    target = torch.tensor(target, dtype=torch.long)
    my_dataset = utils.TensorDataset(data,target)

    data2=torch.from_numpy(b)
    target2=torch.from_numpy(b2)
    target2 = torch.tensor(target2, dtype=torch.long)
    my_testdataset = utils.TensorDataset(data2,target2)


    data2=torch.from_numpy(c)
    target2=torch.from_numpy(c2)
    target2 = torch.tensor(target2, dtype=torch.long)
    my_hortestdataset = utils.TensorDataset(data2,target2)


    train_loader = torch.utils.data.DataLoader(my_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    print("train set loaded" )

    test_loader = torch.utils.data.DataLoader(my_testdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print("test set loaded")

    hortest_loader = torch.utils.data.DataLoader(my_hortestdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print("test set loaded")

    print(" CONVOLUTION NET")
    model = Netconv().to(device)


    print("Net")
    #print(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_no=(np.arange(1,(args.epochs+1),1 ))
    redacc=(np.arange(1,(args.epochs+1),1 ))
    grnacc=(np.arange(1,(args.epochs+1),1 ))

    redaccm=(np.arange(1,(args.epochs+1),1 ))
    grnaccm=(np.arange(1,(args.epochs+1),1 ))

    trnacc=(np.arange(1,(args.epochs+1),1 ))
    trnaccm=(np.arange(1,(args.epochs+1),1 ))
    br=0#test(args, model, device, hortest_loader) #RED IS HORIZONTAL


    bg=0#test(args, model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        [r,g,t,ta]=train(args, model, device, train_loader, optimizer, epoch,hortest_loader,test_loader)
        redaccm[epoch-1]=r
        grnaccm[epoch-1]=g
        trnaccm[epoch-1]=t
        redacc[epoch-1]=test(args, model, device, hortest_loader)
        grnacc[epoch-1]=test(args, model, device, test_loader)
        trnacc[epoch-1]=ta
    resultred = np.empty((redacc.size + redaccm.size+1,),dtype=redacc.dtype)
    resultred[0]=br
    resultred[1::2] = redaccm
    resultred[2::2] = redacc

    resultgrn = np.empty((grnacc.size + grnaccm.size+1,), dtype=grnacc.dtype)
    resultgrn[0]=bg
    resultgrn[1::2] = grnaccm
    resultgrn[2::2] = grnacc

    resulttrn = np.empty((trnacc.size + trnaccm.size+1,),dtype=redacc.dtype)
    resulttrn[0]=0
    resulttrn[1::2] = trnaccm
    resulttrn[2::2] = trnacc
    e=(np.arange(0,(args.epochs+0.5),0.5 ))
    print(e)
    print("resultred",resultred)
    print("resultgrn",resultgrn)
    print("resulttrn",resulttrn)
    plotgraph(e,resultred,resultgrn, resulttrn)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
