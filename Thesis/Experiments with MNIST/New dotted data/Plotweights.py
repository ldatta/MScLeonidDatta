




import numpy as np
import pickle
import gzip
from CKA-Centered-Kernel-Alignment-master import cca_core
from CKA import linear_CKA, kernel_CKA

X = np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/CKA-Centered-Kernel-Alignment-master/Vweight1.npy')
X = np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/CKA-Centered-Kernel-Alignment-master/Hweight1.npy')





print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))















# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Dec  9 12:35:30 2019
# 
# @author: leoniddatta
# """
# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image  
# import numpy as np
# import scipy.ndimage
# import scipy
# import scipy.ndimage.filters as C
# 
# 
# 
# 
# 
# a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/train.npy')  #HORIZONTAL training data 60000*28*28*1
# a2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/trainlabel.npy')
# b=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')   #VERTICAL test data 10000*28*28*1
# b2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy') 
# c=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')  #HORIZONTAL test data
# c2=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/testlabel.npy')
# a=a[:,:,:,0]
# b=b[:,:,:,1]
# c=c[:,:,:,1]
# 
# mask3=np.zeros((28,28))   
# for i in range(0,28,4):
#     mask3[i]=1
#     for j in range(0,28,4):
#         mask3[i,j]=0
# 
# print("mask3 is ")
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
# 
# 
# 
# a=a*mask4   #mask 2 is square #mask is horizontal
# b=b*mask4
# c=c*mask3
# 
# imgh=np.zeros((32,28,28))
# imgv=np.zeros((32,28,28))
# for i in range(32):
#     imgh[i]=C.correlate(b[i,:,:],b[i,:,:], output=None, mode='constant', cval=0.0, origin=0)
# for i in range(32):
#     imgv[i]=C.correlate(c[i,:,:],c[i,:,:], output=None, mode='constant', cval=0.0, origin=0)
# 
# 
# 
# 
# #WITHOUT MAXPOOL
# #img=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/investigate nomaxpool/WITH maxpool/testverticalconv1.npy')
# #imgg=img[8]
# 
# # =============================================================================
# # img=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/investigate nomaxpool/no maxpool/testverticalmaxpooled.npy')
# # imgg=img[8]
# # =============================================================================
# 
# # =============================================================================
# # #WITH MAXPOOL
# # img=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/investigate nomaxpool/with maxpool/testhorizontalmaxpooled.npy')
# # imgg=img[8]
# # =============================================================================
# 
# def printit(a):
#     fig, ((a1, a2,a3,a4,a5,a6,a7,a8),(a9,a10,a11,a12,a13,a14,a15,a16),(a17,a18,a19,a20,a21,a22,a23,a24),(a25,a26,a27,a28,a29,a30,a31,a32)) = plt.subplots(4, 8)
#     #fig.suptitle('Vertical training: vertical test data WITHmaxpool Conv1 ')
#     fig.set_figheight(8)
#     fig.set_figwidth(16)
#     a1.imshow(a[0], cmap='gray',  interpolation='nearest')
#     a2.imshow(a[1], cmap='gray',  interpolation='nearest')
#     a3.imshow(a[2], cmap='gray',  interpolation='nearest')
#     a4.imshow(a[3], cmap='gray',  interpolation='nearest')
#     a5.imshow(a[4], cmap='gray',  interpolation='nearest')
#     a6.imshow(a[5], cmap='gray',  interpolation='nearest')
#     a7.imshow(a[6], cmap='gray',  interpolation='nearest')
#     a8.imshow(a[7], cmap='gray',  interpolation='nearest')
#     a9.imshow(a[8], cmap='gray',  interpolation='nearest')
#     a10.imshow(a[9], cmap='gray',  interpolation='nearest')
#     a11.imshow(a[10], cmap='gray',  interpolation='nearest')
#     a12.imshow(a[11], cmap='gray',  interpolation='nearest')
#     a13.imshow(a[12], cmap='gray',  interpolation='nearest')
#     a14.imshow(a[13], cmap='gray',  interpolation='nearest')
#     a15.imshow(a[14], cmap='gray',  interpolation='nearest')
#     a16.imshow(a[15], cmap='gray',  interpolation='nearest')
#     a17.imshow(a[16], cmap='gray',  interpolation='nearest')
#     a18.imshow(a[17], cmap='gray',  interpolation='nearest')
#     a19.imshow(a[18], cmap='gray',  interpolation='nearest')
#     a20.imshow(a[19], cmap='gray',  interpolation='nearest')
#     a21.imshow(a[20], cmap='gray',  interpolation='nearest')
#     a22.imshow(a[21], cmap='gray',  interpolation='nearest')
#     a23.imshow(a[22], cmap='gray',  interpolation='nearest')
#     a24.imshow(a[23], cmap='gray',  interpolation='nearest')
#     a25.imshow(a[24], cmap='gray',  interpolation='nearest')
#     a26.imshow(a[25], cmap='gray',  interpolation='nearest')
#     a27.imshow(a[26], cmap='gray',  interpolation='nearest')
#     a28.imshow(a[27], cmap='gray',  interpolation='nearest')
#     a29.imshow(a[28], cmap='gray',  interpolation='nearest')
#     a30.imshow(a[29], cmap='gray',  interpolation='nearest')
#     a31.imshow(a[30], cmap='gray',  interpolation='nearest')
#     a32.imshow(a[31], cmap='gray',  interpolation='nearest')
#     plt.show()
# 
# printit(imgh)
# printit(imgv)
# #printit(imgg)
# 
# 
# 
# 
# 
# # =============================================================================
# # a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/conv1.npy')  #HORIZONTAL training data 60000*28*28*1
# # 
# # b=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/max1.npy') 
# # for i in range(5):
# #      plt.imshow(a[5,i], cmap='gray',  interpolation='nearest')
# #      plt.show()
# #      plt.imshow(b[5,i], cmap='gray',  interpolation='nearest')
# #      plt.show()
# # =============================================================================
#      
#      
# # =============================================================================
# #   
# # a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/train.npy')  #HORIZONTAL training data 60000*28*28*1
# # 
# # b=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')   #VERTICAL test data 10000*28*28*1
# # c=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/test.npy')  #HORIZONTAL test data
# # a=a[:,:,:,0]
# # b=b[:,:,:,1]
# # c=c[:,:,:,1]
# # 
# # a2=np.zeros((a.shape[0],56,56))
# # b2=np.zeros((b.shape[0],56,56))
# # c2=np.zeros((c.shape[0],56,56))
# # 
# # for i in range(a.shape[0]):
# #     a2[i] = rescale(a[i], 2, anti_aliasing=True)
# # 
# # for i in range(b.shape[0]):
# #     b2[i] = rescale(b[i], 2, anti_aliasing=True)
# #     c2[i] = rescale(c[i], 2, anti_aliasing=True)
# # 
# # arr = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,1]])
# # mask=np.tile(arr,(14,14))
# # print("mask is")
# # plt.imshow(mask, cmap='gray',  interpolation='nearest')
# # plt.show()
# # a2scaled=a2*mask
# # b2scaled=b2*mask
# # c2scaled=c2*mask
# # 
# # arr = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,1],[0,1,0,1]])
# # mask=np.tile(arr,(14,14))
# # print("mask is")
# # plt.imshow(mask, cmap='gray',  interpolation='nearest')
# # plt.show()
# # 
# # for i in range(3):
# #     plt.imshow(a2scaled[i], cmap='gray',  interpolation='nearest')
# #     plt.show()
# #     plt.imshow(b2scaled[i], cmap='gray',  interpolation='nearest')
# #     plt.show()
# #     plt.imshow(c2scaled[i], cmap='gray',  interpolation='nearest')
# #     plt.show()
# # # =============================================================================
# # =============================================================================
# # for i in range(10):
# #     plt.imshow(b[i], cmap='gray',  interpolation='nearest')
# #     plt.show()
# # =============================================================================
# #a = np.resize(a,(56,56))# , refcheck=False) 
# 
# 
# # =============================================================================
# # a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/visualise filters/trained on vertical data/E7conv1.npy')
# # a=np.reshape(a,(32,3,3))
# # 
# # =============================================================================
# 
# 
# #(a,a,a,a,a,a,a,a)
# 
# 
# 
# 
# 
# 
# # =============================================================================
# # 
# # 
# # a=np.load('/Users/leoniddatta/Dropbox/TU Delft/Study material/DL THESIS/Experiments with MNIST/New dotted data/visualise filters/trained on horizontal data/E1conv1.npy')
# # a=np.reshape(a,(32,3,3))
# # fig, ((ax1, ax2,ax3,ax4,ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax,ax,ax,ax),(ax1, ax2,ax3),(ax1, ax2,ax3)) = plt.subplots(4, 8)
# # fig.suptitle('Filters')
# # fig.set_figheight(4)
# # fig.set_figwidth(6)
# # ax1.imshow(a[0], cmap='gray',  interpolation='nearest')
# # ax2.imshow(a[1], cmap='gray',  interpolation='nearest')
# # ax3.imshow(a[2], cmap='gray',  interpolation='nearest')
# # ax4.imshow(a[3], cmap='gray',  interpolation='nearest')
# # ax5.imshow(a[4], cmap='gray',  interpolation='nearest')
# # ax6.imshow(a[5], cmap='gray',  interpolation='nearest')
# # plt.show()
# # 
# # =============================================================================
# # =============================================================================
# # fig, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5])) = plt.subplots(2, 3)
# # fig.suptitle('Filters')
# # fig.set_figheight(4)
# # fig.set_figwidth(6)
# # ax[0].imshow(a[0], cmap='gray',  interpolation='nearest')
# # ax[1].imshow(a[1], cmap='gray',  interpolation='nearest')
# # ax[2].imshow(a[2], cmap='gray',  interpolation='nearest')
# # ax[3].imshow(a[3], cmap='gray',  interpolation='nearest')
# # ax[4].imshow(a[4], cmap='gray',  interpolation='nearest')
# # ax[5].imshow(a[5], cmap='gray',  interpolation='nearest')
# # plt.show()
# # =============================================================================
# 
# =============================================================================
