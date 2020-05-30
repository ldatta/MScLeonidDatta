from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transform 
from torchvision import datasets, transforms
import torch.utils.data as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import math 

GL=1 #SET GL=0 for Red-7-shaped training Data , Set GL=1 for Green-L-shaped training Data
print("GL =",GL)
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
                          markersize=10, label='RL test data')
    plt.legend(handles=[blue_line,red_line,green_line],loc=2)
    plt.show()
    


class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        st=2
        self.conv1 = nn.Conv2d(3, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, st)
        self.conv3 = nn.Conv2d(8, 16, 3, st)
        self.conv4 = nn.Conv2d(16, 32, 3, st)
        self.conv5 = nn.Conv2d(32, 10, 3, st)
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
    
def train(args, model, device, train_loader, optimizer, epoch, hortest_loader,test_loader):
    r=0
    g=0
    t=0
    running_loss = 0
    total_train = 0
    correct_train = 0
    model.train() 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
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
        modellossmid=0
        if batch_idx % args.log_interval == 0:
            if batch_idx % args.log_interval == 0:
                if (q%10) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \taccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),train_accuracy))
                if (q==50):
                    r,lossr=test(args, model, device, hortest_loader)
                    g,lossg=test(args, model, device, test_loader)
                    t=int(train_accuracy)  
                    modellossmid=float(loss)
        modellossf=float(loss)
    #print(modellossmid,modellossf)
    
    return [r,g,t,train_accuracy,modellossmid,modellossf,lossr,lossg]

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
    test_loss=float(test_loss)
    return acc,test_loss

def main():
    no_epochs=50
    sitr=6
    
    restrn=np.zeros((sitr,(no_epochs*2)+1))
    resgrn=np.zeros((sitr,(no_epochs*2)+1))
    resred=np.zeros((sitr,(no_epochs*2)+1))
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=no_epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    
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
    
    #L SHAPE
    
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
    a[:,:,:,0]=aaa
    b[:,:,:,0]=bbb
    c[:,:,:,0]=ccc
    
#     if(GL==1):
    
#         else:
#             a[:,:,:,0]=aaa
#             b[:,:,:,0]=bbb
#             c[:,:,:,0]=ccc
# =============================================================================
    
    
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
    
    
    print("first train data a[0]=np.",repr(a[0,20:30,20:30,0]))
#     a=np.reshape(a,(60000,1,56,56))
#     b=np.reshape(b,(10000,1,56,56))
#     c=np.reshape(c,(10000,1,56,56))
        
# =============================================================================
    a=np.transpose(a, (0,3, 1, 2))
    b=np.transpose(b, (0,3, 1, 2))
    c=np.transpose(c, (0,3, 1, 2))
# =============================================================================
    
    
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
    for seeditr in range(sitr):
        
#         parser.add_argument('--seed', type=int, default=seeditr+1, metavar='S',
#                         help='random seed (default: 1)')
        torch.manual_seed(seeditr+1)
    
    
        train_loader = torch.utils.data.DataLoader(my_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
        print("train set loaded" )
    
        test_loader = torch.utils.data.DataLoader(my_testdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
        print("test set loaded")
    
        hortest_loader = torch.utils.data.DataLoader(my_hortestdataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
        print("test set loaded")
    
        print(" CONVOLUTION NET")
        model = Netconv().to(device)
        #torch.save(model.state_dict(), "May27R7sameWIni.pt")
        print("Net")
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        epoch_no=(np.arange(1,(args.epochs+1),1 ))
        redacc=(np.arange(1,(args.epochs+1),1 ))
        grnacc=(np.arange(1,(args.epochs+1),1 ))
    
        redaccm=(np.arange(1,(args.epochs+1),1 ))
        grnaccm=(np.arange(1,(args.epochs+1),1 ))
    
        trnacc=(np.arange(1,(args.epochs+1),1 ))
        trnaccm=(np.arange(1,(args.epochs+1),1 ))
        
        lossmtrn=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        lossftrn=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        
        lossmred=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        lossfred=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        
        lossmgrn=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        lossfgrn=(np.arange(1,(args.epochs+1),1 ,dtype=float))
        br=0#test(args, model, device, hortest_loader) #RED IS HORIZONTAL
           
           
        bg=0#test(args, model, device, test_loader)
        for epoch in range(1, args.epochs + 1):    
            [r,g,t,ta,midls,ls,lossr,lossg]=train(args, model, device, train_loader, optimizer, epoch,hortest_loader,test_loader)
            redaccm[epoch-1]=r
            grnaccm[epoch-1]=g
            trnaccm[epoch-1]=t
            lossmtrn[epoch-1]=midls
            lossmred[epoch-1]=lossr
            lossmgrn[epoch-1]=lossg
            redacc[epoch-1],lossfred[epoch-1]=test(args, model, device, hortest_loader)
            grnacc[epoch-1],lossfgrn[epoch-1]=test(args, model, device, test_loader)
            trnacc[epoch-1]=ta
            lossftrn[epoch-1]=ls
        resultred = np.empty((redacc.size + redaccm.size+1,),dtype=redacc.dtype)
        resultred[0]=0
        resultred[1::2] = redaccm
        resultred[2::2] = redacc
    
        resultgrn = np.empty((grnacc.size + grnaccm.size+1,), dtype=grnacc.dtype)
        resultgrn[0]=0
        resultgrn[1::2] = grnaccm
        resultgrn[2::2] = grnacc
    
        resulttrn = np.empty((trnacc.size + trnaccm.size+1,),dtype=redacc.dtype)
        resulttrn[0]=0
        resulttrn[1::2] = trnaccm
        resulttrn[2::2] = trnacc
        
        
        lossarraytrn = np.empty((lossftrn.size + lossmtrn.size+1,),dtype=lossftrn.dtype)
        lossarraytrn[0]=0
        lossarraytrn[1::2] = lossmtrn
        lossarraytrn[2::2] = lossftrn
        
        lossarrayred = np.empty((lossfred.size + lossmred.size+1,),dtype=lossfred.dtype)
        lossarrayred[0]=0
        lossarrayred[1::2] = lossmred
        lossarrayred[2::2] = lossfred
        
        lossarraygrn = np.empty((lossfgrn.size + lossmgrn.size+1,),dtype=lossfgrn.dtype)
        lossarraygrn[0]=0
        lossarraygrn[1::2] = lossmgrn
        lossarraygrn[2::2] = lossfgrn
        
        lossarraytrn[0]=lossarraytrn[1]
        lossarrayred[0]=lossarrayred[1]
        lossarraygrn[0]=lossarraygrn[1]
        
        
        e=(np.arange(0,(args.epochs+0.5),0.5 ))
        print("SeedItr",seeditr+1)
        print(e)
        print("resultred",repr(resultred))
        print("resultgrn",repr(resultgrn))
        print("resulttrn",repr(resulttrn))
        #print("lossarraytrn",repr(lossarraytrn))
        #print("lossarrayred",repr(lossarrayred))
        #print("lossarraygrn",repr(lossarraygrn))
        restrn[seeditr]=resulttrn
        resred[seeditr]=resultred
        resgrn[seeditr]=resultgrn
        #if args.save_model:
        print("Running Base RL, tested on R7 and RL ")
        torch.save(model.state_dict(), 'BaseMay31RLseed{}.pt'.format(seeditr+1))
        print("model saved")
    print("average Base RL, tested on R7 and RL ")
    print("restrn=np.",repr(restrn))
    print("resred=np.",repr(resred))
    print("resgrn=np.",repr(resgrn))
    np.save('May31baseRLtrn.npy',restrn)
    np.save('May31baseRLred.npy',resred)
    np.save('May31baseRLgrn.npy',resgrn)
    print("average res ")
    print("restrn=np.",repr(np.mean(restrn, axis=0)))
    print("resred=np.",repr(np.mean(resred, axis=0)))
    print("resgrn=np.",repr(np.mean(resgrn, axis=0)))
    
    
    
    
if __name__ == '__main__':
    main()
