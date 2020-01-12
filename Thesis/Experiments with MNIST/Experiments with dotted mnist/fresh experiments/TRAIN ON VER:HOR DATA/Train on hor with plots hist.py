#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:49:01 2019

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
import numpy as np
import tensorflow as tf
import torch.utils.data as utils
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from collections import namedtuple
import tensorflow as tf
import tensorflow.summary
from torch.utils.tensorboard import SummaryWriter
from tensorflow.summary import scalar
from tensorflow.summary import histogram
from chardet.universaldetector import UniversalDetector
#TRAIN FILE 'horTrainData.npy'
#TEST FILE 

a=np.load('horTrainData.npy')  #HORIZONTAL training data 60000*28*28*1
a2=np.load('trainlabel.npy')
b=np.load('verTestData.npy')   #VERTICAL test data 10000*28*28*1
b2=np.load('testlabel.npy') 
c=np.load('horTestData.npy')  #HORIZONTAL test data
c2=np.load('testlabel.npy')
# =============================================================================
# k=5000
# a=a[0:k,:,:]
# b=b[0:k,:,:]
# c=c[0:k,:,:]
# a2=a2[0:k]
# b2=b2[0:k]
# c2=c2[0:k]
# =============================================================================
cora=np.array(np.unique(a2, return_counts=True))[1]

corb=np.array(np.unique(b2, return_counts=True))[1]

corc=np.array(np.unique(c2, return_counts=True))[1]
# =============================================================================
# for i in range (a.shape[0]):
#     a[i,:,:]=np.rot90(a[i,:,:],2)
# 
# for i in range (b.shape[0]):
#     b[i,:,:]=np.rot90(b[i,:,:],2)
# 
# for i in range (c.shape[0]):
#     c[i,:,:]=np.rot90(c[i,:,:],2)
#     
# =============================================================================
# =============================================================================
# for i in range(200):
#     if a2[i]==4:
#         plt.imshow(a[i], cmap='gray', interpolation='nearest')
#         plt.show()
#         
# for i in range(200):
#     if a2[i]==3:
#         plt.imshow(a[i], cmap='gray', interpolation='nearest')
#         plt.show()
#         
# for i in range(200):
#     if a2[i]==7:
#         plt.imshow(a[i], cmap='gray', interpolation='nearest')
#         plt.show()
#     
#      
# =============================================================================

print("train data is")
for i in range (2):
        plt.imshow(a[i], cmap='gray', interpolation='nearest')
        plt.show()
        
print("Horizontal test data is")
for i in range (2):
        plt.imshow(c[i], cmap='gray',  interpolation='nearest')
        plt.show()
print("Vertical test data is")
for i in range (2):
        plt.imshow(b[i], cmap='gray', interpolation='nearest')
        plt.show()
a=np.reshape(a,(60000,1,28,28))
b=np.reshape(b,(10000,1,28,28))
c=np.reshape(c,(10000,1,28,28))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 10, 3, 1)
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
        
    def forward(self, x):
        x=x.float()
        x=self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = self.GAP(x)
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x        
        

def plotgraph (xs,y1s,y2s,yts):
    plt.clf()
    
    fig = plt.figure(figsize=(10, 7))
    plt.plot(xs,y1s,'s:r')
    fig.suptitle('TEST ACCURACY GRAPH')
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
    plt.show()       
    
def train(args, model, device, train_loader, optimizer, epoch):
    r=0
    g=0
    
    running_loss = 0
    total_train = 0
    correct_train = 0
    full_prediction_label = np.array ([])
    full_target = np.array ([])
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
        
        full_prediction_label=np.concatenate ((full_prediction_label, predicted.numpy()))
        full_target=np.concatenate ((full_target, target.numpy()))
        
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
                    
# =============================================================================
#                 if (q==50):
#                     r=test(args, model, device, hortest_loader)
#                     g=test(args, model, device, test_loader)
#                     t=int(train_accuracy)
# =============================================================================
    #t=train_accuracy
    t=int(train_accuracy)
    wrong=np.zeros((10))
    right=np.zeros((10))
    #print(predic.shape,"predic shape")
    m=np.array([0,1,2,3,4,5,6,7,8,9])
    #print ("output Shape",full_target.shape)
    #print(full_prediction_label.shape,"full prediction")
    for i in range(60000):
        for j in range (10):
            if full_target[i]==j:
                if full_target[i]!=full_prediction_label[i]:
                    wrong[j]=wrong[j]+1
                    #wrongnumber=np.concatenate((wrongnumber,m[j]))
    
    for i in range(60000):
        for j in range (10):
            if full_target[i]==j:
                if full_target[i]==full_prediction_label[i]:
                    right[j]=right[j]+1
    fig = plt.figure(figsize=(10, 7))
    plt.plot(m,right,'sr')
    fig.suptitle('Correct predictions')
    plt.xlabel('Training - symbol: RED predicted correct label,  GREEN actual correct label ')
    plt.ylabel('Number of correct predictions')
    for x,y in zip(m,right):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')  
    plt.plot(m,cora,'^g')
    for x,y in zip(m,cora):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
    plt.show() 
   
    return [r,g,t,train_accuracy]


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    full_prediction_label = np.array ([])
    full_target = np.array ([])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predicted=pred.numpy()
            predicted=np.reshape(predicted,(predicted.shape[0]))
            full_prediction_label=np.concatenate ((full_prediction_label, predicted))
            full_target=np.concatenate ((full_target, target.numpy()))
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc=int(100. * correct / len(test_loader.dataset))
    wrong=np.zeros((10))
    right=np.zeros((10))
    #print(predic.shape,"predic shape")
    m=np.array([0,1,2,3,4,5,6,7,8,9])
    #print ("output Shape",full_target.shape)
    #print(full_prediction_label.shape,"full prediction")
    for i in range(len(test_loader.dataset)):
        for j in range (10):
            if full_target[i]==j:
                if full_target[i]!=full_prediction_label[i]:
                    wrong[j]=wrong[j]+1
                    #wrongnumber=np.concatenate((wrongnumber,m[j]))
    
    for i in range(len(test_loader.dataset)):
        for j in range (10):
            if full_target[i]==j:
                if full_target[i]==full_prediction_label[i]:
                    right[j]=right[j]+1
    fig = plt.figure(figsize=(10, 7))
    plt.plot(m,right,'sr')
    fig.suptitle('Correct predictions')
    plt.xlabel('Testing - symbol: RED predicted correct label,  GREEN actual correct label ')
    plt.ylabel('Number of correct predictions')
    for x,y in zip(m,right):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')  
    plt.plot(m,corb,'^g')
    for x,y in zip(m,corb):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
    plt.show() 
    
    return acc




    
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
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



model = Net().to(device)
print("Net")
writer = None 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
epoch_no=(np.arange(1,(args.epochs+1),1 ))
redacc=(np.arange(1,(args.epochs+1),1 ))
grnacc=(np.arange(1,(args.epochs+1),1 ))

redaccm=(np.arange(1,(args.epochs+1),1 ))
grnaccm=(np.arange(1,(args.epochs+1),1 ))

trnacc=(np.arange(1,(args.epochs+1),1 ))
trnaccm=(np.arange(1,(args.epochs+1),1 ))
#bt=test(args, model, device, train_loader)
br=0#test(args, model, device, hortest_loader) #RED IS HORIZONTAL
   
   
bg=0#test(args, model, device, test_loader)
for epoch in range(1, args.epochs + 1):    
    [r,g,t,ta]=train(args, model, device, train_loader, optimizer, epoch)
    redaccm[epoch-1]=r
    grnaccm[epoch-1]=g
    trnaccm[epoch-1]=t
    
    redacc[epoch-1]=0#test(args, model, device, hortest_loader)
   
    grnacc[epoch-1]=0#test(args, model, device, test_loader)
    trnacc[epoch-1]=ta
#plt.plot(epoch_no,redacc)  
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

plotgraph(e,resultred,resultgrn, resulttrn)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
