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
import scipy.ndimage.filters as C
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math 

GL=1

k=14
k2=14
k3=14
def sortit(a):
    a=a.detach().cpu().numpy() 
    z=np.zeros((a.shape))
    for i in range (a.shape[0]):
        matrix_mean_list = []
        for k in a:
            x = np.mean(k)
            matrix_mean_list.append((k, x))

        matrix_mean_list = sorted(matrix_mean_list, key=lambda m: m[1])
        n=np.asarray(matrix_mean_list)
        #print(n.shape)
        #print(n[0][0].shape)
        #z=np.zeros((a.shape))
        for j in range(a.shape[0]):
            z[j]=n[j][0]
    z=torch.from_numpy(z)
    return z

def npsave(resultred,resultgrn,resulttrn):
    np.save('GLRGBsortred.npy',resultred)
    np.save('GLRGBsortgrn.npy',resultgrn)
    np.save('GLRGBsorttrn.npy',resulttrn)
    print("hello GL=1")
#     np.save('R7RGBsortred.npy',resultred)
#     np.save('R7RGBsortgrn.npy',resultgrn)
#     np.save('R7RGBsorttrn.npy',resulttrn)
#     print("hello GL=0")
    
    
    
def weightit(in_channel,outc,k,g):
    range_value=in_channel*k*k
    range_value=1. / math.sqrt(range_value)
    
    weight=np.random.uniform(-range_value,range_value,k*k)
    weight=np.reshape(weight,(k,k))
    
        
    if (g==1):
        weightf=np.zeros(( outc,in_channel,k,k))
        for i in range(outc):
            for j in range(in_channel):
                weightf[i,j,:,:]=weight
        
    else:
        
        weightf=np.zeros((outc,k,k))
        
        for j in range(outc):
            weightf[j,:,:]=weight
        weightf=np.reshape(weightf,(outc,1,k,k))
    return weightf

class NetconvDep(nn.Module):
    def __init__(self):
        super(NetconvDep, self).__init__()
        st=2
        st1=2
        self.conv1 = nn.Conv2d(3, k*3, 3, 1, groups=3)
        self.conv11 = nn.Conv2d(k*3, k2*16, 1, 1)
        self.conv2 = nn.Conv2d(k2*16, k*16, 3, st1,groups=k3*16)
        self.conv22 = nn.Conv2d(k*16, k2*32, 1, st)
        self.conv3 = nn.Conv2d(k2*32, k*32, 3, st1,groups=k3*32)
        self.conv33 = nn.Conv2d(k*32, 10, 1, st)
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
        
    
    def forward(self, x):
        x=x.float()
        x=self.conv1(x) 
        x = F.relu(x)
        x=sortit(x)
        x=x.float()
        x=x.cuda()
        x=self.conv11(x) 
        x = F.relu(x)
        x=self.conv2(x) 
        x = F.relu(x)
        x=sortit(x)
        x=x.float()
        x=x.cuda()
        x=self.conv22(x) 
        x = F.relu(x)
        x=self.conv3(x) 
        x = F.relu(x)
        x=sortit(x)
        x=x.float()
        x=x.cuda()
        x=self.conv33(x) 
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
    weight1=weightit(1, k*3, 3,3)
    weight11=weightit(k*3, k2*16, 1,1)
    weight2=weightit(k2*16, k*16, 3,k3*16)
    weight22=weightit(k*16, k2*32, 1,1)
    weight3=weightit(k2*32, k*32, 3,k3*32)
    weight33=weightit(k*32, 10, 1,1)
    weight1=np.float32(weight1)
    weight11=np.float32(weight11)
    weight2=np.float32(weight2)
    weight22=np.float32(weight22)
    weight3=np.float32(weight3)
    weight33=np.float32(weight33)
#     print("CONVOLUTION NET")
#     model = Netconv().to(device)
    weight1=torch.from_numpy(weight1).to(device)
    weight11=torch.from_numpy(weight11).to(device)
    weight2=torch.from_numpy(weight2).to(device)
    weight22=torch.from_numpy(weight22).to(device)
    weight3=torch.from_numpy(weight3).to(device)
    weight33=torch.from_numpy(weight33).to(device)
    model.conv1.weight.data=weight1
    model.conv11.weight.data=weight11
    model.conv2.weight.data=weight2
    model.conv22.weight.data=weight22
    model.conv3.weight.data=weight3
    model.conv33.weight.data=weight33
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output.shape)
        
        #print(target.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
# =============================================================================
#         weight1 = model.conv1.weight.data.numpy()
#         weight11 = model.conv11.weight.data.numpy()
#         weight2 = model.conv2.weight.data.numpy()
#         weight22 = model.conv22.weight.data.numpy()
#         weight3 = model.conv3.weight.data.numpy()
#         weight33 = model.conv33.weight.data.numpy()
# =============================================================================
# =============================================================================
#         print(weight1.shape,"weight 1 ")
#         print(weight11.shape,"weight 11")
#         print(weight2.shape,"weight 2")
#         print(weight22.shape,"weight 22 ")
#         print(weight3.shape,"weight 3")
#         print(weight33.shape,"weight 33")
# =============================================================================
        
# =============================================================================
#         #np.save('L1.npy',weight1)
#         weight2 = model.conv2.weight.data.numpy()
#         np.save('LtrainedWeight2.npy',weight2)
#         weight3 = model.conv3.weight.data.numpy()
#         np.save('LtrainedWeight3.npy',weight3)
#         weight4 = model.conv4.weight.data.numpy()
#         np.save('LtrainedWeight4.npy',weight4)
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
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=130, metavar='N',
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
    print(a.shape)
    print(a[0].max())
    print(a[0].min())
    
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
        
    a=np.transpose(a, (0,3, 1, 2))
    b=np.transpose(b, (0,3, 1, 2))
    c=np.transpose(c, (0,3, 1, 2))
#     a=np.reshape(a,(60000,1,datasize,datasize))
#     b=np.reshape(b,(10000,1,datasize,datasize))
#     c=np.reshape(c,(10000,1,datasize,datasize))

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

    print(" DEPTHWISE CONVOLUTION NET")
    model = NetconvDep().to(device)
    
    
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
    #plotgraph(e,resultred,resultgrn, resulttrn)# ,bresultred,bresultgrn, bresulttrn)
    npsave(resultred,resultgrn,resulttrn)
#     np.save('notun22GLRGBKK2K313maskgap5red.npy',resultred)
#     np.save('notun22GLRGBKK2K313maskgap5grn.npy',resultgrn)
#     np.save('notun22GLRGBKK2K313maskgap5trn.npy',resulttrn)
    
# #     np.save('notun22R7RGBKK2K313maskgap5red.npy',resultred)
# #     np.save('notun22R7RGBKK2K313maskgap5grn.npy',resultgrn)
# #     np.save('notun22R7RGBKK2K313maskgap5trn.npy',resulttrn)
    
#     #bresultred=np.load('Baseresults/INLtrainedresultred.npy')
#     #bresultgrn=np.load('Baseresults/INLtrainedresultgrn.npy')  
#     #bresulttrn=np.load('Baseresults/INLtrainedresulttrn.npy')
                       
#     plotgraph(e,resultred,resultgrn, resulttrn)#,bresultred,bresultgrn, bresulttrn)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
if __name__ == '__main__':
    main()
