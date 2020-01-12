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
#TRAIN FILE 'horTrainData.npy'
#TEST FILE 

a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/visualise filters/rtraindata.npy')  #HORIZONTAL training data 60000*28*28*1
a2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/trainlabel.npy')
b=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/visualise filters/rtestdata1.npy')   #VERTICAL test data 10000*28*28*1
b2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy') 
c=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/visualise filters/rtestdata2.npy')   #VERTICAL test data 10000*28*28*1
c2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy')

# =============================================================================
# arr = np.array([[0,1,1,1,0,0],[1,1,1,1,1,0],[ 1,1,0,1,1,0],[1,1,1,1,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0 ]])
# mask=np.tile(arr,(5,5))
# mask=mask[0:28,0:28]
# 
# # =============================================================================
# # print("mask is")
# # plt.imshow(mask, cmap='gray', interpolation='nearest')
# # plt.show()
# # =============================================================================
# # =============================================================================
# # 
# # mask2=np.ones((28,28))
# # for i in range(1,28,5):
# #     mask2[i,:]=0
# #     mask2[:,i]=0
# # 
# # print("mask2 is")
# # plt.imshow(mask2, cmap='gray', interpolation='nearest')
# # plt.show()
# # 
# # =============================================================================
# mask3=np.zeros((28,28))   
# for i in range(0,28,4):
#     mask3[i]=1
#     for j in range(0,28,4):
#         mask3[i,j]=0
# 
# print("mask3 is")
# plt.imshow(mask3, cmap='gray', interpolation='nearest')
# plt.show()
# 
# mask4=np.zeros((28,28))   
# for i in range(0,28,4):
#     mask4[i]=1
#     for j in range(0,28,4):
#         mask4[i,j]=0
#         mask4[i,j]=0
# mask4=mask4.T
# print("mask4 is")
# plt.imshow(mask4, cmap='gray', interpolation='nearest')
# plt.show()
# 
# =============================================================================
maskarray1 = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,1]])


mask1=np.tile(maskarray1,(14,14))
for i in range(0,56,4):
     for j in range(0,56,4):
         mask1[i,j]=0
 


print("mask1 is")
plt.imshow(mask1, cmap='gray',  interpolation='nearest')
plt.show()

maskarray2 = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,1],[0,1,0,1]])
mask2=np.tile(maskarray2,(14,14))
print("mask2 is")
plt.imshow(mask2, cmap='gray',  interpolation='nearest')
plt.show()
# =============================================================================
# 
# maskarray3 = np.array([[0,0,0,0],[0,1,0,1],[0,0,1,0],[0,1,0,1]])
# mask3=np.tile(maskarray3,(14,14))
# print("mask3 is")
# plt.imshow(mask3, cmap='gray',  interpolation='nearest')
# =============================================================================
plt.show()

a=a*mask1   #mask 2 is square #mask is horizontal
b=b*mask1
c=c*mask2

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



a=np.reshape(a,(60000,1,a.shape[1],a.shape[1]))
b=np.reshape(b,(10000,1,b.shape[1],b.shape[1]))
c=np.reshape(c,(10000,1,c.shape[1],c.shape[1]))

def printit(img):
    print("the filters are")
    img=img.data.numpy()
    for i in range(img.shape[1]):
        print(img[1,i,:,:].shape)
        
        plt.imshow(img[1,i,:,:], cmap='gray', interpolation='nearest')
        plt.show()
        
    return img
        



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, 3, stride=2)
        self.conv2 = nn.Conv2d(25, 22, 3, stride=2)
        self.conv3 = nn.Conv2d(22, 10, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 10, 3, stride=1)
        self.GAP=nn.AvgPool2d((11,11), stride=1, padding=0)
        
    def forward(self, x):
        x=x.float()
        x=self.conv1(x)
        #print("shape after CONV1",x.shape)
        x = F.relu(x)
        #x = F.max_pool2d(x,2, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        #print("shape after CONV2",x.shape)
        x=self.conv3(x)
        x = F.relu(x)
        #print("shape after CONV3",x.shape)
        #x = F.max_pool2d(x,2, 2)
        #x=self.conv4(x)
        #x = F.relu(x)
        #print("shape after CONV4",x.shape)
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
        #weightconv1 = model.conv1.weight.data.numpy()
        #weightconv2 = model.conv2.weight.data.numpy()
        #weightconv3 = model.conv3.weight.data.numpy()
        #print(weight.shape)
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
    #t=int(train_accuracy)    
    #wconv1=weightconv1
    #wconv2=weightconv2
    #wconv3=weightconv3
    return [r,g,t,train_accuracy]


def test(args, model, device, test_loader):
    #model.eval()
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
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
epoch_no=(np.arange(1,(args.epochs+1),1 ))
redacc=(np.arange(1,(args.epochs+1),1 ))
grnacc=(np.arange(1,(args.epochs+1),1 ))

redaccm=(np.arange(1,(args.epochs+1),1 ))
grnaccm=(np.arange(1,(args.epochs+1),1 ))

trnacc=(np.arange(1,(args.epochs+1),1 ))
trnaccm=(np.arange(1,(args.epochs+1),1 ))
br=0#test(args, model, device, hortest_loader) #RED IS HORIZONTAL
#wconv1=np.empty(args.epochs,32,1,3,3)
#wconv2=np.empty(args.epochs,64,1,3,3)
   
bg=0#test(args, model, device, test_loader)
for epoch in range(1, args.epochs + 1):    
    [r,g,t,ta]=train(args, model, device, train_loader, optimizer, epoch)
    redaccm[epoch-1]=r
    grnaccm[epoch-1]=g
    trnaccm[epoch-1]=t
    
    #wconv1[epoch-1]=wcnv1
    #wconv2[epoch-1]=wcnv2
    
    redacc[epoch-1]=test(args, model, device, hortest_loader)
   
    grnacc[epoch-1]=test(args, model, device, test_loader)
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
