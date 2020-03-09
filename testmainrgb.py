
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
import tensorflow as tf
import torch.utils.data as utils

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from attention import AttentionConv, AttentionStem
from rgbfunctions import plotgraph, train, test

att=0
red=0

class NetconvwithoutBN(nn.Module):
    def __init__(self):
        super(NetconvwithoutBN, self).__init__()
        print("NetconvwithoutBN")
        self.conv1 = nn.Conv2d(3, 16, 3, 1,padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1,padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1,padding=2)
        self.conv4 = nn.Conv2d(64, 128, 3, 1,padding=2)
        self.conv5 = nn.Conv2d(128, 10, 3, 1,padding=2)
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
        self.m = nn.AvgPool2d((2, 2),stride=(2,2))
        
        
    def forward(self, x):
        x=x.float()
        x=self.conv1(x) 
        x = self.bn1(x)
        x = F.relu(x)
        x=self.m(x)
        #print(x.shape,"after conv 1")
        #s1=x.data.numpy()
        #x = F.relu( self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x=self.m(x)
        #s2=x.data.numpy()
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv3(x))
        x=self.m(x)
        #print(x.shape,"after conv 3")
        #s3=x.data.numpy()
        #x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv4(x))
        #print(x.shape,"after conv 3")
        
        x=self.m(x)
        #print(x.shape,"after conv 3")
        x = F.relu(self.conv5(x))
        #x = F.relu(self.bn5(self.conv5(x)))
        #s4=x.data.numpy()
        x=self.m(x)
        #print(x.shape, "before gap")
        x = self.GAP(x)
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
    
    
class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        print("convolution net")
        self.conv1 = nn.Conv2d(3, 16, 3, 1,padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1,padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1,padding=2)
        self.conv4 = nn.Conv2d(64, 128, 3, 1,padding=2)
        self.conv5 = nn.Conv2d(128, 10, 3, 1,padding=2)
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
        self.m = nn.AvgPool2d((2, 2),stride=(2,2))
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.bn4=nn.BatchNorm2d(128)    
        self.bn5=nn.BatchNorm2d(10) 
        
    def forward(self, x):
        x=x.float()
        x=self.conv1(x) 
        x = self.bn1(x)
        x = F.relu(x)
        x=self.m(x)
        #print(x.shape,"after conv 1")
        #s1=x.data.numpy()
        x = F.relu( self.bn2(self.conv2(x)))
        x=self.m(x)
        #s2=x.data.numpy()
        x = F.relu(self.bn3(self.conv3(x)))
        x=self.m(x)
        #print(x.shape,"after conv 3")
        #s3=x.data.numpy()
        x = F.relu(self.bn4(self.conv4(x)))
        #print(x.shape,"after conv 3")
        
        x=self.m(x)
        #print(x.shape,"after conv 3")
        x = F.relu(self.bn5(self.conv5(x)))
        #s4=x.data.numpy()
        x=self.m(x)
        #print(x.shape, "before gap")
        x = self.GAP(x)
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
    
class NetAtt(nn.Module):
    def __init__(self):
        super(NetAtt, self).__init__()
    
        print("attention net")
        self.att1 =nn.Sequential(AttentionStem(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=2, groups=1))
        self.att2 =nn.Sequential(AttentionStem(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2, groups=1))
        self.att3 =nn.Sequential(AttentionStem(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, groups=1))
        self.att4 =nn.Sequential(AttentionStem(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=2, groups=1))
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
        self.m = nn.AvgPool2d((2, 2),stride=(2,2))
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.bn4=nn.BatchNorm2d(10)    
    def forward(self, x):
        x=x.float()
        #print(x.shape,"SIZE BEFORE NETWORK STARTS" )
        x=self.att1(x) 
        x = self.bn1(x)
        x = F.relu(x)
        x=self.m(x)
        #x = F.max_pool2d(x,2, 2)
        #print(x.shape,"SIZE AFTER ATTENTION 1" )
        x=self.att2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x=self.m(x)
        #print(x.shape,"SIZE AFTER ATTENTION 2" )
        x=self.att3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x=self.m(x)
        #print(x.shape,"SIZE AFTER ATTENTION 3" )
        x=self.att4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x=self.m(x)
        #print(x.shape,"SIZE AFTER ATTENTION 4" )
        x = self.GAP(x)
        #print(x.shape,"SIZE AFTER GAP" )
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
                              



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
    
    a22=np.zeros((60000,56,56))
    b22=np.zeros((10000,56,56))
    c22=np.zeros((10000,56,56))
    
    for i in range (aa.shape[0]):
        a22[i]=cv2.resize(aa[i], (56, 56))
        
    
    for i in range (bb.shape[0]):
        b22[i]=cv2.resize(bb[i], (56, 56))
        
    
    for i in range (cc.shape[0]):
        c22[i]=cv2.resize(cc[i], (56, 56))
    a22=a22/255
    b22=b22/255
    c22=c22/255
    
    datasize=56
    
    
    a=np.zeros((60000,56,56,3))
    b=np.zeros((10000,56,56,3))
    c=np.zeros((10000,56,56,3))
    
    if(red==0):
        print("GREEN TRAIN DATA") #For green train data 
        a[:,:,:,1]=a22
    
    if(red==1):
        print("RED TRAIN DATA") #For red train data
        a[:,:,:,0]=a22
    
    
    
    b[:,:,:,0]=b22
    c[:,:,:,1]=c22
      
    print("train data is")
    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Training Data (blue curve train accyracy in graph)')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(a[0],    interpolation='nearest')
    ax2.imshow(a[1],    interpolation='nearest')
    ax3.imshow(a[2],    interpolation='nearest')
    ax4.imshow(a[3],    interpolation='nearest')
    ax5.imshow(a[4],    interpolation='nearest')
    ax6.imshow(a[5],    interpolation='nearest')
    plt.show()


    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Test data1 - green curve accuracy in graph')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(b[0],    interpolation='nearest')
    ax2.imshow(b[1],    interpolation='nearest')
    ax3.imshow(b[2],    interpolation='nearest')
    ax4.imshow(b[3],    interpolation='nearest')
    ax5.imshow(b[4],    interpolation='nearest')
    ax6.imshow(b[5],    interpolation='nearest')
    plt.show()

    fig, ((ax1, ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3)
    fig.suptitle('Test data2 - red curve accuracy in graph')
    fig.set_figheight(7)
    fig.set_figwidth(10)
    ax1.imshow(c[0],    interpolation='nearest')
    ax2.imshow(c[1],    interpolation='nearest')
    ax3.imshow(c[2],    interpolation='nearest')
    ax4.imshow(c[3],    interpolation='nearest')
    ax5.imshow(c[4],    interpolation='nearest')
    ax6.imshow(c[5],    interpolation='nearest')
    plt.show()

    a=np.reshape(a,(60000,3,datasize,datasize))
    b=np.reshape(b,(10000,3,datasize,datasize))
    c=np.reshape(c,(10000,3,datasize,datasize))

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

    
    NetconvwithoutBN
    
    print("NetconvwithoutBN  56 x 56 NET")
    model = NetconvwithoutBN().to(device)
    
#     if(att==0):
#       print("CONVOLUTION  56 x 56 NET")
#       model = Netconv().to(device)
    
#     if(att==1):
#       print("ATTENTION NET")
#       model = NetAtt().to(device)
    
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
    plotgraph(e,resultred,resultgrn, resulttrn)# ,bresultred,bresultgrn, bresulttrn)
    
#     if((att!=1) and (red==1)):
#         print("Convo model Red Data")
#         np.save('convRedtrainedresultred.npy',resultred)
#         np.save('convRedtrainedresultgrn.npy',resultgrn)
#         np.save('convRedtrainedresulttrn.npy',resulttrn)
#     if((att!=1) and (red!=1)):
#         print("Convo model Green Data")
#         np.save('convGrntrainedresultred.npy',resultred)
#         np.save('convGrntrainedresultgrn.npy',resultgrn)
#         np.save('convGrntrainedresulttrn.npy',resulttrn)
#     if((att==1) and (red==1)):
#         print("Attention model Red Data")
#         np.save('AttRedtrainedresultred.npy',resultred)
#         np.save('AttRedtrainedresultgrn.npy',resultgrn)
#         np.save('AttRedtrainedresulttrn.npy',resulttrn)
#     if((att==1) and (red!=1)):
#         print("Attention model Greeen Data")
#         np.save('AttGrntrainedresultred.npy',resultred)
#         np.save('AttGrntrainedresultgrn.npy',resultgrn)
#         np.save('AttGrntrainedresulttrn.npy',resulttrn)
    
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
if __name__ == '__main__':
    main()




