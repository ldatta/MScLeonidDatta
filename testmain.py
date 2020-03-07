
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
import cv2
import tensorflow as tf
import torch.utils.data as utils
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from attention import AttentionConv, AttentionStem





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
                          markersize=10, label='INL test data')
    green_line = mlines.Line2D([], [], color='green', marker='^',
                          markersize=10, label='L test data')
    plt.legend(handles=[blue_line,red_line,green_line],loc=2)
    plt.show()   
    
class Net28(nn.Module):
    def __init__(self):
        super(Net28, self).__init__()
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


class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 10, 3, 2)
        self.GAP=nn.AvgPool2d((5,5), stride=1, padding=0)
        
    def forward(self, x):
        x=x.float()
        x=self.conv1(x) 
        x = F.relu(x)
        s1=x.data.numpy()
        x = F.relu(self.conv2(x))
        s2=x.data.numpy()
        x = F.relu(self.conv3(x))
        s3=x.data.numpy()
        x = F.relu(self.conv4(x))
        s4=x.data.numpy()
        x = self.GAP(x)
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
    

class NetAtt(nn.Module):
    def __init__(self):
        super(NetAtt, self).__init__()
    
        #print("hcfcfcgfcgf")
        self.att1 =nn.Sequential(AttentionStem(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2, groups=1))
        self.att2 =nn.Sequential(AttentionStem(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2, groups=1))
        self.att3 =nn.Sequential(AttentionStem(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, groups=1))
        self.att4 =nn.Sequential(AttentionStem(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=2, groups=1))
        self.GAP=nn.AvgPool2d((1,1), stride=1, padding=0)
            
    def forward(self, x):
        x=x.float()
        #print(x.shape,"SIZE BEFORE NETWORK STARTS" )
        x=self.att1(x) 
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        #print(x.shape,"SIZE AFTER ATTENTION 1" )
        x=self.att2(x) 
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        #print(x.shape,"SIZE AFTER ATTENTION 2" )
        x=self.att3(x) 
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        #print(x.shape,"SIZE AFTER ATTENTION 3" )
        x=self.att4(x) 
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        #print(x.shape,"SIZE AFTER ATTENTION 4" )
        x = self.GAP(x)
        #print(x.shape,"SIZE AFTER GAP" )
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x

                              
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
# =============================================================================
#         if (epoch==20):
#             weight1 = model.conv1.weight.data.numpy()
#             np.save('LtrainedWeight1.npy',weight1)
#             weight2 = model.conv2.weight.data.numpy()
#             np.save('LtrainedWeight2.npy',weight2)
#             weight3 = model.conv3.weight.data.numpy()
#             np.save('LtrainedWeight3.npy',weight3)
#             weight4 = model.conv4.weight.data.numpy()
#             np.save('LtrainedWeight4.npy',weight4)
# =============================================================================
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
                    r=0#test(args, model, device, hortest_loader)
                    g=0#test(args, model, device, test_loader)
                    t=int(train_accuracy)   
    
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
            #np.save('predlabelLtrainedINLtest.npy',pred)
            #print(pred.shape,"predicted test")
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
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
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
    
    datasize=56
    print(a.shape)

    a=1*(a>0.5)
    b=1*(b>0.5)
    c=1*(c>0.5)
    mask=np.zeros((datasize,datasize))

    for i in range(0,datasize,4):
        for j in range(0,datasize,4):
            mask[i,j]=1 
    
    
# =============================================================================
#     for i in range (aa.shape[0]):
#         a[i]=a[i]*mask
#         
#     
#     for i in range (bb.shape[0]):
#         b[i]=b[i]*mask
#         
#     
#     for i in range (cc.shape[0]):
#         c[i]=c[i]*mask
# =============================================================================
        
        
    a=a*mask
    b=b*mask
    c=c*mask

    #L SHAPE
    # =============================================================================
    # for k in range(a.shape[0]):
    #     for i in range(0,datasize,4):
    #         for j in range(0,datasize,4):
    #             if(a[k,i,j]==1):
    #                 a[k,i,j]=0
    #                 a[k,i,j-1]=1
    #                 a[k,i-1,j-1]=1
    #                 a[k,i+1,j-1]=1
    #                 a[k,i+1,j]=1
    # =============================================================================


    #INVERSE L
    for k in range(a.shape[0]):
        for i in range(0,datasize,4):
            for j in range(0,datasize,4):
                if(a[k,i,j]==1):
                    a[k,i,j]=0
                    a[k,i-1,j]=1
                    a[k,i-1,j+1]=1
                    a[k,i,j+1]=1
                    a[k,i+1,j+1]=1
                    

    for k in range(b.shape[0]):
        for i in range(0,datasize,4):
            for j in range(0,datasize,4):
                if(b[k,i,j]==1):
                    b[k,i,j]=0
                    b[k,i,j-1]=1
                    b[k,i-1,j-1]=1
                    b[k,i+1,j-1]=1
                    b[k,i+1,j]=1
                    
    for k in range(c.shape[0]):
        for i in range(0,datasize,4):
            for j in range(0,datasize,4):
                if(c[k,i,j]==1):
                    c[k,i,j]=0
                    c[k,i-1,j]=1
                    c[k,i-1,j+1]=1
                    c[k,i,j+1]=1
                    c[k,i+1,j+1]=1


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
        
    a=np.reshape(a,(60000,1,datasize,datasize))
    b=np.reshape(b,(10000,1,datasize,datasize))
    c=np.reshape(c,(10000,1,datasize,datasize))

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


    #print("CONVOLUTION NET")
    #model = Netconv().to(device)
    
    #print("CONVOLUTION NET for 28x28")
    #model = Net28().to(device)
    
    print("ATTENTION NET")
    model = NetAtt().to(device)



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
        [r,g,t,ta]=train(args, model, device, train_loader, optimizer, epoch)
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

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
if __name__ == '__main__':
    main()




