


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

#from attention import AttentionConv, AttentionStem


def cor(img,img2):
    #sum1=np.zeros((img.shape[0],28,28))
    img=img.cpu().numpy()
    img2=img2.cpu().numpy()
    cor=np.zeros((img.shape))#[0],img.shape[1],28,28))
    for i in range (img.shape[0]):
        #print("image shape",img.shape,img.shape[0])
        for j in range (img.shape[1]):
            cor[i,j,:,:]=C.correlate(img[i,j,:,:],img2[i,j,:,:], output=None, mode='constant', cval=0.0, origin=0)

    cor=torch.from_numpy(cor).float().cuda()
    return cor

add=64

k=8
k2=8
class NetconvDepw2(nn.Module):
    def __init__(self):
        super(NetconvDepw2, self).__init__()
        st=1
        st1=2
        self.conv1 = nn.Conv2d(1, k*1, 3, 1, groups=1)
        self.conv11 = nn.Conv2d(k*1, k2*16, 1, 1)
        self.conv2 = nn.Conv2d(k2*16, k*16, 3, st1,groups=16)
        self.conv22 = nn.Conv2d(k*16, k2*32, 1, st)
        self.conv3 = nn.Conv2d(k2*32, k*32, 3, st1,groups=32)
        self.conv33 = nn.Conv2d(k*32, k2*64, 1, st)
        self.conv4 = nn.Conv2d(k2*64, k*64, 3, st1,groups=64)
        self.conv44 = nn.Conv2d(k*64, k2*128, 1, st)
        self.conv5 = nn.Conv2d(k2*128, k*128, 3, st1,groups=128)
        self.conv55 = nn.Conv2d(k*128, 10, 1, st)
        self.pool=nn.AvgPool2d((2,2), stride=2, padding=0)
        self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
               
        
    def forward(self, x):
        x=x.float()
        x=F.relu(self.conv1(x) )
        x=self.conv11(x) 
        x = F.relu(x)
        #x=self.pool(x)
        #x = F.max_pool2d(x,2, 2)
        #s1=x.data.numpy()
        x = F.relu(self.conv22(F.relu(self.conv2(x))))
        
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #s2=x.data.numpy()
        #print(x.shape,"conv3 ")
        x=F.relu(self.conv3(x))
        #print(x.shape,"conv3 ")
        x = F.relu(self.conv33(x))
        #print(x.shape,"conv3 ")
        #s3=x.data.numpy()
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #print(x.shape)
        x = F.relu(self.conv44(F.relu(self.conv4(x))))
        #s4=x.data.numpy()
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #x=add_channel(x,add)
        #x=x.float()
        #print(x.shape)
        x = F.relu(self.conv55(F.relu(self.conv5(x))))
        #s5=x.data.numpy()
        #x = F.max_pool2d(x,2, 2)
#         print(x.shape, "before gap")
        x = self.GAP(x)
        
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
class NetconvDep(nn.Module):
    def __init__(self):
        super(NetconvDep, self).__init__()
        st=2
        st1=2
        self.conv1 = nn.Conv2d(1, k*1, 3, 1, groups=1)
        self.conv11 = nn.Conv2d(k*1, k2*16, 1, 1)
        self.conv2 = nn.Conv2d(k2*16, k*16, 3, st1,groups=16)
        self.conv22 = nn.Conv2d(k*16, k2*32, 1, st)
        self.conv3 = nn.Conv2d(k2*32, k*32, 3, st1,groups=32)
        self.conv33 = nn.Conv2d(k*32, 10, 1, st)
        self.conv4 = nn.Conv2d(k2*64, k*64, 3, st1,groups=64)
        self.conv44 = nn.Conv2d(k*64, 10, 1, st)
        self.conv5 = nn.Conv2d(128, k*128, 3, st1,groups=128)
        self.conv55 = nn.Conv2d(k*128, 10, 1, st)
        self.pool=nn.AvgPool2d((2,2), stride=2, padding=0)
        self.GAP=nn.AvgPool2d((3,3), stride=1, padding=0)
               
        
    def forward(self, x):
        x=x.float()
        x=F.relu(self.conv1(x) )
        x=self.conv11(x) 
        x = F.relu(x)
        #x=self.pool(x)
        #x = F.max_pool2d(x,2, 2)
        #s1=x.data.numpy()
        x = F.relu(self.conv22(F.relu(self.conv2(x))))
        
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #s2=x.data.numpy()
        #print(x.shape,"conv3 ")
        x=F.relu(self.conv3(x))
        #print(x.shape,"conv3 ")
        x = F.relu(self.conv33(x))
        #print(x.shape,"conv3 ")
        #s3=x.data.numpy()
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #print(x.shape)
        #x = F.relu(self.conv44(self.conv4(x)))
        #s4=x.data.numpy()
        #x=self.pool(x)#x = F.max_pool2d(x,2, 2)
        #x=add_channel(x,add)
        #x=x.float()
        #print(x.shape)
        #x = F.relu(self.conv55(self.conv5(x)))
        #s5=x.data.numpy()
        #x = F.max_pool2d(x,2, 2)
        #print(x.shape, "before gap")
        x = self.GAP(x)
        
        x = x.view(-1, 10) 
        x=F.log_softmax(x, dim=1)
        return x
    
    
# print("L train data - Layer 5 updating")                      
def train(args, model, device, train_loader, optimizer, epoch, hortest_loader,test_loader):
    r=0
    g=0
    t=0
    running_loss = 0
    total_train = 0
    correct_train = 0
    model.train() 
#     model.conv1.weight.data = torch.from_numpy(INL1)
#     model.conv2.weight.data = torch.from_numpy(INL2)
#     model.conv3.weight.data = torch.from_numpy(INL3)
#     model.conv4.weight.data = torch.from_numpy(INL4)
    #model.conv5.weight.data = torch.from_numpy(INL5)
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
                    r=test(args, model, device, hortest_loader)
                    g=test(args, model, device, test_loader)
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
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    
    maskgap=6
    
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
    
    plt.imshow(b[0], cmap='gray',  interpolation='nearest')
    plt.show()
    
    a=a*mask
    b=b*mask
    c=c*mask
    
    
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
        
        

    #L SHAPE
#     for k in range(a.shape[0]):
#         for i in range(0,datasize,maskgap):
#             for j in range(0,datasize,maskgap):
#                 if(a[k,i,j]==1):
#                     a[k,i,j]=0
#                     a[k,i,j-1]=1
#                     a[k,i-1,j-1]=1
#                     a[k,i+1,j-1]=1
#                     a[k,i+1,j]=1
     
    
    #INVERSE L

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

    print(" DEPTHWISE CONVOLUTION NET")
    model = NetconvDep().to(device)

#     print("CONVOLUTION NET")
#     model = Netconv().to(device)
    
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
    np.save('INLtrainWorkingred.npy',resultred)
    np.save('INLtrainWorkinggrn.npy',resultgrn)
    np.save('INLtrainWorkingtrn.npy',resulttrn)
    
    #bresultred=np.load('Baseresults/INLtrainedresultred.npy')
    #bresultgrn=np.load('Baseresults/INLtrainedresultgrn.npy')  
    #bresulttrn=np.load('Baseresults/INLtrainedresulttrn.npy')
                       
    plotgraph(e,resultred,resultgrn, resulttrn)#,bresultred,bresultgrn, bresulttrn)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
if __name__ == '__main__':
    main()




