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

#Testing on R7 and GL data


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

tune_size=2000

btune=b[0:tune_size]
btest = b[tune_size:,:]
b2tune=b2[0:tune_size]
b2test = b2[tune_size:]

ctune=c[0:tune_size]
ctest = c[tune_size:,:]
c2tune=c2[0:tune_size]
c2test = c2[tune_size:]



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

btune=np.transpose(btune, (0,3, 1, 2))
ctune=np.transpose(ctune, (0,3, 1, 2))

btest=np.transpose(btest, (0,3, 1, 2))
ctest=np.transpose(ctest, (0,3, 1, 2))
  

        
bdata2tune=torch.from_numpy(btune)
btarget2tune=torch.from_numpy(b2tune)
btarget2tune = torch.tensor(btarget2tune, dtype=torch.long)
my_testdatasettune = utils.TensorDataset(bdata2tune,btarget2tune)

cdata2tune=torch.from_numpy(ctune)
ctarget2tune=torch.from_numpy(c2tune)
ctarget2tune = torch.tensor(ctarget2tune, dtype=torch.long)
my_hortestdatasettune = utils.TensorDataset(cdata2tune,ctarget2tune)


bdata2test=torch.from_numpy(btest)
btarget2test=torch.from_numpy(b2test)
btarget2test = torch.tensor(btarget2test, dtype=torch.long)
my_testdatasettest = utils.TensorDataset(bdata2test,btarget2test)

cdata2test=torch.from_numpy(ctest)
ctarget2test=torch.from_numpy(c2test)
ctarget2test = torch.tensor(ctarget2test, dtype=torch.long)
my_hortestdatasettest = utils.TensorDataset(cdata2test,ctarget2test)




grntest_loadertune = torch.utils.data.DataLoader(my_testdatasettune,batch_size=args.test_batch_size, shuffle=True, **kwargs)
redtest_loadertune = torch.utils.data.DataLoader(my_hortestdatasettune,batch_size=args.test_batch_size, shuffle=True, **kwargs)

grntest_loadertest = torch.utils.data.DataLoader(my_testdatasettest,batch_size=args.test_batch_size, shuffle=True, **kwargs)
redtest_loadertest = torch.utils.data.DataLoader(my_hortestdatasettest,batch_size=args.test_batch_size, shuffle=True, **kwargs)




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



#print("Layer No is",layer_no)
in_c2=4
in_c3=8
in_c4=16
in_c5=32

finallist=[]
#R7acc=np.zeros((in_c,in_c))

seed_nos=10


#for seed_no in range(10):
for seed_no in range(10):
    listtune=[]
    listtest=[]
    maxtillnow=11
    model = Netconv()
    model.load_state_dict(torch.load('GLmodels/BaseGLseed{seed}.pt'.format(seed=seed_no+1)))
    model.conv1.weight[:,0,:,:]=model.conv1.weight[:,0,:,:]+model.conv1.weight[:,1,:,:]
    model.conv1.weight[:,1,:,:]=model.conv1.weight[:,0,:,:]-model.conv1.weight[:,1,:,:]
    model.conv1.weight[:,0,:,:]=model.conv1.weight[:,0,:,:]-model.conv1.weight[:,1,:,:]
    model.cuda()
    maxtillnow=test(args, model, device, redtest_loadertune)
    print("New Seed No is",seed_no+1)
    for itr in range(5):
        for i2 in range (in_c2):
            for j2 in range(in_c2):
                
                if(i2>=j2):
                    if(i2!=j2):
                      
                        model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]+model.conv2.weight[:,j2,:,:]
                        model.conv2.weight[:,j2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                        model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                    print("Iter no",itr+1,"seed no",seed_no+1,"Checking Conv 2 channel no",i2,j2,"max till now",maxtillnow)
                    model.cuda()
                    accconv2=test(args, model, device, redtest_loadertune)
                    accconv2test=test(args, model, device, redtest_loadertest)
                    listtune.append(accconv2)
                    listtest.append(accconv2test)
                    if(accconv2<maxtillnow):
                        if (i2!=j2):
                            model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]+model.conv2.weight[:,j2,:,:]
                            model.conv2.weight[:,j2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                            model.conv2.weight[:,i2,:,:]=model.conv2.weight[:,i2,:,:]-model.conv2.weight[:,j2,:,:]
                            print("not better. swaping back",i2,"and",j2 )
                    if(accconv2>maxtillnow):
                        maxtillnow=accconv2  
        
        
        
        for i3 in range (in_c3):
            for j3 in range(in_c3):
                if(i3>=j3):
                    if(i3!=j3):
                        model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]+model.conv3.weight[:,j3,:,:]
                        model.conv3.weight[:,j3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                        model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                    print("Iter no",itr+1,"seed no",seed_no+1,"Checking Conv 3 channel no",i3,j3)
                    model.cuda()
                    accconv3=test(args, model, device, redtest_loadertune)
                    accconv3test=test(args, model, device, redtest_loadertest)
                    listtune.append(accconv3)
                    listtest.append(accconv3test)
                    if(accconv3<maxtillnow):
                        if(i3!=j3):
                            model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]+model.conv3.weight[:,j3,:,:]
                            model.conv3.weight[:,j3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                            model.conv3.weight[:,i3,:,:]=model.conv3.weight[:,i3,:,:]-model.conv3.weight[:,j3,:,:]
                            print("not better. swaping back",i3,"and",j3 )
                    if(accconv3>maxtillnow):
                        maxtillnow=accconv3    
                            
                            
                      
                        
        for i4 in range (in_c4):
            for j4 in range(in_c4):
                if(i4>=j4):
                    if(i4!=j4):
                        model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]+model.conv4.weight[:,j4,:,:]
                        model.conv4.weight[:,j4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                        model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                    print("Iter no",itr+1,"seed no",seed_no+1,"Checking Conv 4 channel no",i4,j4,"max till now",maxtillnow)
                    model.cuda()
                    accconv4=test(args, model, device, redtest_loadertune)
                    accconv4test=test(args, model, device, redtest_loadertest)
                    listtune.append(accconv4)
                    listtest.append(accconv4test)
                    if(accconv4<maxtillnow):
                        if(i4!=j4):
                            model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]+model.conv4.weight[:,j4,:,:]
                            model.conv4.weight[:,j4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                            model.conv4.weight[:,i4,:,:]=model.conv4.weight[:,i4,:,:]-model.conv4.weight[:,j4,:,:]
                            print("not better. swaping back",i4,"and",j4 )
                    if(accconv4>maxtillnow):
                        maxtillnow=accconv4  
                    
        for i5 in range (in_c5):
            for j5 in range(in_c5):
                
                if(i5>=j5):
                    if(i5!=j5):
                        model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]+model.conv5.weight[:,j5,:,:]
                        model.conv5.weight[:,j5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                        model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                    print("Iter no",itr,"seed no",seed_no+1,"Checking Conv 5 channel no",i5,j5,"max till now",maxtillnow)
                    model.cuda()
                    accconv5=test(args, model, device, redtest_loadertune)
                    accconv5test=test(args, model, device, redtest_loadertest)
                    listtune.append(accconv5)
                    listtest.append(accconv5test)
                    if(accconv5<maxtillnow):
                        if(i5!=j5):
                            model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]+model.conv5.weight[:,j5,:,:]
                            model.conv5.weight[:,j5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                            model.conv5.weight[:,i5,:,:]=model.conv5.weight[:,i5,:,:]-model.conv5.weight[:,j5,:,:]
                            print("not better. swaping back",i5,"and",j5 )
                    if(accconv5>maxtillnow):
                        maxtillnow=accconv5  
        finallist.append(maxtillnow)
    
    np.save('GLtrainedalwaysKeepBetterotequalactualR7seed{seed}Itr5tune_size2000tunelist'.format(seed=seed_no+1),listtune)
    np.save('GLtrainedalwaysKeepBetterotequalactualR7seed{seed}Itr5tune_size2000testlist'.format(seed=seed_no+1),listtest)



print(finallist)
print("Tune Size 2000")

