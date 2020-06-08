#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:24:23 2020

@author: leoniddatta
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



#for conv_no in range
# =============================================================================
# a7=np.load('ExchangeResults/R7trained/R7trnConv5inpexResR7.npy')
# aL=np.load('ExchangeResults/R7trained/R7trnConv5inpexResRL.npy')
# print("R7 max",a7[conv].max())
# print("RL max",aL[conv].max())
# =============================================================================
# =============================================================================
# 
# seed=4
# 
# print("Conv 2")
# a7=np.load('ExchangeResults/RLtrained/RLtrnConv2inpexResR7.npy')
# aL=np.load('ExchangeResults/RLtrained/RLtrnConv2inpexResRL.npy')
# print(a7[seed-1])
# print("R7 max",a7[seed-1].max())
# print("RL max",aL[seed-1].max())
# 
# print("Conv 3")
# a7=np.load('ExchangeResults/RLtrained/RLtrnConv3inpexResR7.npy')
# aL=np.load('ExchangeResults/RLtrained/RLtrnConv3inpexResRL.npy')
# print(a7[seed-1])
# print("R7 max",a7[seed-1].max())
# print("RL max",aL[seed-1].max())
# 
# print("Conv 4")
# a7=np.load('ExchangeResults/RLtrained/RLtrnConv4inpexResR7.npy')
# aL=np.load('ExchangeResults/RLtrained/RLtrnConv4inpexResRL.npy')
# print(a7[seed-1])
# print("R7 max",a7[seed-1].max())
# print("RL max",aL[seed-1].max())
# 
# print("Conv 5")
# a7=np.load('ExchangeResults/RLtrained/RLtrnConv5inpexResR7.npy')
# aL=np.load('ExchangeResults/RLtrained/RLtrnConv5inpexResRL.npy')
# print(a7[seed-1])
# print("R7 max",a7[seed-1].max())
# print("RL max",aL[seed-1].max())
# 
# 
# 
# 
# 
# 
# =============================================================================




def plotgraph (xs,y1s,y2s,yts,y1smax,y2smax,ytsmax,y1smin,y2smin,ytsmin):
    plt.clf()
    fig = plt.figure(figsize=(11, 8))
    plt.plot(xs,y1s,'s:r')
    plt.fill_between(xs, y1smax, y1smin,facecolor='red',alpha=0.1)
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
    plt.fill_between(xs, y2smax, y2smin,facecolor='green',alpha=0.1)
    for x,y in zip(xs,y2s):
        label = "{:.0f}".format(y)
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    plt.plot(xs,yts,'*:b')
    plt.fill_between(xs, ytsmax, ytsmin,facecolor='blue',alpha=0.1)
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
                          markersize=10, label='GL test data')
    plt.legend(handles=[blue_line,red_line,green_line],loc=2)
    plt.show()
e=(np.arange(0,(40+0.5),0.5 ))

restrn=np.load('GLbaserestrnseed1to10.npy')
resgrn=np.load('GLbaseresgrnseed1to10.npy')
resred=np.load('GLbaseresredseed1to10.npy')
#resgrn=(resgrn*0)+11


plotgraph(e,np.mean(resred, axis=0),np.mean(resgrn, axis=0),np.mean(restrn, axis=0),np.max(resred, axis=0),np.max(resgrn, axis=0),np.max(restrn, axis=0),np.min(resred, axis=0),np.min(resgrn, axis=0),np.min(restrn, axis=0)) 




#plotgraph(e,np.mean(resred, axis=0),np.mean(resgrn, axis=0),np.mean(restrn, axis=0),np.std(resred, axis=0),np.std(resgrn, axis=0),np.std(restrn, axis=0)) #resred,resgrn, np.mean(restrn, axis=0))


















