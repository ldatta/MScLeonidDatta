
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
import scipy.ndimage.filters as C
#import FlowNetPytorch.models.util as corr


  #.models.util.correlate as corr #import correlate
 #FlowNetPytorch.models.util
#import spatial_correlation_sampler_backend as correlation
a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/train.npy')  #red training data 60000*28*28*3
a2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/trainlabel.npy')  #red training data labels
b=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')   #green test data 10000*28*28*3
b2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy') #green testing data labela
# =============================================================================
# a=a[0:5000,:,:,:]
# a2=a2[0:5000]
# =============================================================================
c=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')   #green test data 10000*28*28*3
c2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy') #green testing data labela

c[:,:,:,0]=c[:,:,:,1]
c[:,:,:,1]=np.zeros((c[:,:,:,1].shape))
#c is the red test data

print("train data")
for i in range (2):
    plt.imshow(a[i], interpolation='nearest')
    plt.show()
print("test data 1")
for i in range (2):
    plt.imshow(b[i], interpolation='nearest')
    plt.show()
print("test data 2")
for i in range (2):
    plt.imshow(c[i], interpolation='nearest')
    plt.show()
    
a=np.transpose(a,(0,3, 1, 2))  #from 28*28*3, transposing to 3*28*28
b=np.transpose(b,(0,3, 1, 2))   
c=np.transpose(c,(0,3, 1, 2)) 
def cor(img):
    #sum1=np.zeros((img.shape[0],28,28))
    img=img.detach().numpy()
    #img2=img[:,:,5:22,5:22]  #K=17
    img2=img[:,2,8:19,8:19] #K=2k+1=11
    #img2=img[:,:,8:19,8:19]
    #img2=img[:,:,8:19,8:19]
    cor1=np.zeros((img.shape[0],28,28))#[0],img.shape[1],28,28))
    for i in range (img.shape[0]):
        #print("image shape",img.shape,img.shape[0])
        #for j in range (img.shape[1]):
        cor1[i,:,:]=C.correlate(img[i,2,:,:],img2[i,:,:], output=None, mode='constant', cval=0.0, origin=0)
    cor1=np.reshape(cor1,(img.shape[0],1,28,28))
    cor1=torch.from_numpy(cor1)
    
   # print(cor1.shape)
    return cor1  
# =============================================================================
# 
# Network: 
#     conv1
#     relu
#     maxpool
#     cov2
#     relu 
#     maxpool
#     conv3
#     GAP
#     softmax
# 
# =============================================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 10, 3, 1)
        #self.correlate=(out_conv3a,out_conv3b)
        self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
        
    def forward(self, x):
        x=x.float()
        x=cor(x)
        x=x.float()
        x=self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2, 2)
        #print(x.shape, "after maxpl 1")
        x = F.relu(self.conv2(x))
        #print(x.shape, "after conv 2")
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape, "after maxpl 2")
        x = F.relu(self.conv3(x))
        #print(x.shape, "after conv 4")
        x = self.GAP(x)
        #print(x.shape, "after GAP")
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
    #plt.xticks(np.arange(0,100,1))
    #plt.yticks(np.arange(0,10,1))
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
    r=50
    g=60
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
                    r=test(args, model, device, redtest_loader)
                    g=test(args, model, device, test_loader)
                    t=int(train_accuracy)
        
    return [r,g,t,train_accuracy]


def test(args, model, device, test_loader):
    model.eval()
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
parser.add_argument('--epochs', type=int, default=7, metavar='N',
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

#print("train set loaded" )

data2=torch.from_numpy(b)
target2=torch.from_numpy(b2)
target2 = torch.tensor(target2, dtype=torch.long)
my_testdataset = utils.TensorDataset(data2,target2)
#print("test set loaded")

data2=torch.from_numpy(c)
target2=torch.from_numpy(c2)
target2 = torch.tensor(target2, dtype=torch.long)
my_redtestdataset = utils.TensorDataset(data2,target2)
#print("Red test set loaded")
# =============================================================================
# train_y = torch.stack([torch.Tensor(i) for i in a2]) # transform to torch tensors
# test_y = torch.stack([torch.Tensor(i) for i in b2])
# =============================================================================
 # create your datset
#my_dataloader = utils.DataLoader(my_dataset)




train_loader = torch.utils.data.DataLoader(my_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
print("train set loaded" )

test_loader = torch.utils.data.DataLoader(my_testdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
print("test set loaded")

redtest_loader = torch.utils.data.DataLoader(my_redtestdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
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
br=test(args, model, device, redtest_loader)
   
   
bg=test(args, model, device, test_loader)
for epoch in range(1, args.epochs + 1):    
    [r,g,t,ta]=train(args, model, device, train_loader, optimizer, epoch)
    redaccm[epoch-1]=r
    grnaccm[epoch-1]=g
    trnaccm[epoch-1]=t
    #epoch_no[(2*epoch)-1]=epoch
    
    redacc[epoch-1]=test(args, model, device, redtest_loader)
   
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
#resultred=resultred.astype(int)


plotgraph(e,resultred,resultgrn, resulttrn)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
        
# =============================================================================
# if __name__ == '__main__':
#     main()
# =============================================================================
