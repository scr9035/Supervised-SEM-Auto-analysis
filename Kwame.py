# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:22:07 2017

@author: LiDo
"""

import numpy as np
import skimage.external.tifffile as read_tiff
import matplotlib.pyplot as plt
import tkinter
import tkinter.filedialog
from scipy.optimize import curve_fit
from skimage import exposure

import os

from SSA.analysis.GeneralProcess import (PixDistribution, GaussianMixThres, 
                        BinaryConverter, BinaryDialateEdge, BinaryErosionEdge,
                        AnisotropicImageFilter1D, AnisotropicFilter2D, 
                        AnisotropicFilter1D, LGSemEdge, KernelThresh, )
from diffuse import EdgeEnhance2D, ReguIsoNonlinear

def OneDChannel(profile, scale=0.5, target='dark'):
    '''
    target : str, 'dark' or 'bright'
        The center would be the center of dark or bright channel
    '''
    thres = GaussianMixThres(profile, components=2, scale=scale)
    if target == 'dark':
        bi_line = profile > thres
    elif target == 'bright':
        bi_line = profile < thres
    else:
        return None, None, None
    edges = np.diff(bi_line.astype(int))
    up = np.argwhere(edges == 1).flatten()
    down = np.argwhere(edges == -1).flatten()
    x_lim = len(profile)
    
    half_period = int((np.mean(np.diff(up)) + np.mean(np.diff(down)))/4)
    
    if len(up) - len(down) == 1:
        plateau = (up[:-1] + down)/2
        channel_center = (up[1:] + down)/2
        if channel_center[-1] + half_period < x_lim:
            plateau = np.append(plateau, [channel_center[-1]+half_period])
        else:
            channel_center = np.delete(channel_center, -1)
    elif len(down) - len(up) == 1:
        plateau = (up + down[1:])/2
        channel_center = (up + down[:-1])/2
        if channel_center[0] > half_period:
            plateau = np.append([channel_center[0]-half_period], plateau)
        else:
            channel_center = np.delete(channel_center, 0)
    elif len(up) == len(down):
        if up[0] < down[0]:
            plateau = (up + down) / 2
            channel_center = (down[:-1] + up[1:]) / 2
        else:           
            plateau = (up[:-1] + down[1:]) / 2
            channel_center = (up + down) / 2
            if channel_center[0] > half_period:
                plateau = np.append([channel_center[0] - half_period], plateau)
            else:
                channel_center = np.delete(channel_center, 0)
            if channel_center[-1] + half_period < x_lim:                
                plateau = np.append(plateau, [channel_center[-1]+half_period])
            else:
                channel_center = np.delete(channel_center, -1)   
    channel_count = len(channel_center)
    return channel_count, channel_center, plateau

def LogiPlateau(x, x0, y0, k1, c, height, k2, left, right):

    y = c / (1 + np.exp(-k1*(x - x0))) + y0 + height / (1 + np.exp(-k2 * (x-left)) + np.exp(k2*(x-right)))
    return y

def LPSemEdge(x, y, threshold=95, finess=0.05, orientation='forward', plot=False):
    """
    finess :
        amount of pixel
    """
    peak = np.argmax(y)
    peak_intens = y[peak]
    
    
    if orientation == 'forward':
        ori = 1
        y0 = y[0]
        c = y[-1]-y[0]
        
        if peak != 0:
            mid = x[np.argmin(np.abs((y[:peak] - (y[0] + y[-1])/2)))]
        else:
            mid = x[np.argmin(np.abs((y - (y[0] + y[-1])/2)))]
         
#        left_0 = x[np.argmin(np.abs(y[:peak] - (peak_intens + y[-1]) / 2))]
        left_0 = mid
        right_0 = x[peak:][np.argmin(np.abs(y[peak:] - (peak_intens + y[-1]) / 2))]
        height_0 = peak_intens - y[-1]
        
    elif orientation == 'backward':
        ori = -1
        y0 = y[-1]
        c = y[0]-y[-1]

        if peak != (len(y) - 1):
            mid = x[peak:][np.argmin(np.abs((y[peak:] - (y[0] + y[-1])/2)))]
        else:
            mid = x[np.argmin(np.abs((y - (y[0] + y[-1])/2)))]
        
        left_0 = x[np.argmin(np.abs(y[:peak] - (peak_intens + y[0]) / 2))]
        right_0 = mid
#        right_0 = x[peak:][np.argmin(np.abs(y[peak:] - (peak_intens + y[0]) / 2))]
        height_0 = peak_intens - y[0]
        
    p0 = [mid, y0, ori, c, height_0, 1, left_0, right_0]
    bounds = ([np.min(x), 0, -np.inf, 0, 0, 0, np.min(x), np.min(x)],
             [np.max(x), np.max(y), np.inf, np.inf, np.inf, np.inf, np.max(x), np.max(x)])
    try:
        popt, pcov = curve_fit(LogiPlateau, x, y, p0=p0, bounds=bounds)
        N = int((x[-1] - x[0])/finess + 1)
        coord = np.linspace(x[0], x[-1], num=N)
        fx = LogiPlateau(coord, *popt)
        peak = np.argmax(fx)
        
        low_edge = popt[0] - np.log(100.0/threshold-1) / popt[2]
        if orientation == 'forward':
            thres = (np.max(fx) - fx[-1]) * threshold / 100 + fx[-1]
            hi_edge = coord[np.argmin(np.abs(fx[:peak+1] - thres))]

        elif orientation == 'backward':
            thres = (np.max(fx) - fx[0]) * threshold / 100 + fx[0]
            hi_edge = coord[np.argmin(np.abs(fx[peak:] - thres)) + peak]
        edge = 0.5 * (low_edge + hi_edge)
        if plot:
            plt.plot(x, y)
            plt.plot(x, popt[4]/(1 + np.exp(- popt[5] * (x-popt[6])) + np.exp(popt[5] * (x-popt[7]))))
            plt.plot(coord, LogiPlateau(coord, *p0))
            plt.plot(coord, fx)
            plt.plot([edge, edge], [np.min(y), np.max(y)])
            plt.show()     
    except Exception as e:
        print(str(e))
        return None
    return edge

# In[]
# Read images

root = tkinter.Tk()
path = tkinter.filedialog.askopenfilenames(parent=root,title='Choose a files')
root.destroy()
pic_name = os.path.basename(path[0])

image = read_tiff.imread(path)
part = image[:,:800]
#part = image[0,:700,:]
y_lim, x_lim = part.shape

left = image[:800, :500]
f=plt.figure(figsize=(10,8))
plt.imshow(left, cmap=plt.cm.gray)
plt.show()

a, b = exposure.histogram(left,nbins=512)
plt.plot(b,a)
plt.show()

smooth = EdgeEnhance2D(left, 300, 30, sigma=2, delta_t=0.1)
f=plt.figure(figsize=(10,8))
plt.imshow(smooth, cmap=plt.cm.gray)
plt.show()


thres = 110
bi_img = BinaryConverter(smooth, thres=thres)
edge = BinaryDialateEdge(bi_img)
y, x = np.nonzero(edge)

a, b = exposure.histogram(smooth, nbins=512)
plt.plot(b,a)
plt.show()

f=plt.figure(figsize=(10,8))
plt.imshow(left, cmap=plt.cm.gray)
plt.plot(x, y, 'r')
plt.show()


#profile = np.sum(left, axis=1)/500.0
#kappa = 0.2 * (np.max(profile) - np.min(profile))
#smooth = ReguIsoNonlinear(profile, 300, kappa, sigma=5, delta_t=0.1)
#f = plt.figure(figsize=(10,10))
#plt.plot(profile, '.')
#plt.plot(smooth)
#plt.show()
#
#gradient = np.gradient(smooth)
#kappa_grad = 0.2 * (np.max(gradient) - np.min(gradient))
#smooth_grad = ReguIsoNonlinear(gradient, 300, kappa_grad, sigma=5, delta_t=0.1)
#grad_pos_thres = np.max(smooth_grad) * 0.2
#grad_neg_thres = np.min(smooth_grad) * 0.2
#plt.plot(smooth_grad, label='gradient')
#plt.show()
#
#hess = np.gradient(gradient)
#kappa_hess = 0.2 * (np.max(hess) - np.min(hess))
#smooth_hess = ReguIsoNonlinear(hess, 300, kappa_hess, sigma=5, delta_t=0.1)
#plt.plot(smooth_hess, label='hess')
#plt.show()
#
#up = []
#down = []
#mask = (smooth_hess > 0).astype(int)
#crossing = np.nonzero(mask[1:] - mask[:-1])[0]
#for idx in crossing:
#    if smooth_grad[idx] > grad_pos_thres:
#        up.append(idx)
#    if smooth_grad[idx] < grad_neg_thres:
#        down.append(idx)
        
#f = plt.figure(figsize=(10,10))
#plt.imshow(left, cmap=plt.cm.gray)
#plt.hlines(up, 0, 500, 'r', label='up')
#plt.hlines(down, 0, 500, 'b', label='down')
#plt.show()

#up = np.array(up)
#down = np.array(down)
#if len(up) == len(down):
#    if up[0] < down[0]:
#        center = (up + down) / 2
#        plateau = (down[:-1] + up[1:]) / 2       
#    else:
#        plateau = (up[:-1] + down[1:]) / 2
#        channel_center = (up + down) / 2
#elif len(up) - len(down) == 1:
#    center = (up[:-1] + down) / 2
#    plateau = (up[1:] + down) / 2
#elif len(down) - len(up) == 1:
#    center = (up + down[1:]) /2
#    plateau = (up + down[:-1]) / 2
#
#
#f = plt.figure(figsize=(10,10))
#plt.imshow(left, cmap=plt.cm.gray)
#plt.hlines(center, 0, 500, 'r', label='center')
#plt.hlines(plateau, 0, 500, 'b', label='plateau')
#plt.legend(loc=2)
#plt.show()
#
#avg = 10
#edge_center =[]
#edge_plateau = []
#for idx in center:
#    profile = np.sum(left[int(idx):int(idx)+avg,:], axis=0)/avg
#    kappa = 0.2 * (np.max(profile) - np.min(profile))
#    profile = ReguIsoNonlinear(profile, 300, kappa, sigma=5, delta_t=0.1)
#    edge_center.append(LPSemEdge(np.arange(500), profile, threshold=85, 
#                                 finess=0.05, orientation='backward', plot=True))
#for idx in plateau:
#    profile = np.sum(left[int(idx):int(idx)+avg,:], axis=0)/avg
##    kappa = 0.2 * (np.max(profile) - np.min(profile))
##    profile = ReguIsoNonlinear(profile, 100, kappa, sigma=5, delta_t=0.1)
#    edge_plateau.append(LGSemEdge(np.arange(500), profile, threshold=85, 
#                                  finess=0.05, orientation='backward', plot=False))
#
#f = plt.figure(figsize=(10,10))
#plt.imshow(left, cmap=plt.cm.gray)
#plt.scatter(edge_center, center)
#plt.scatter(edge_plateau, plateau)
#plt.show()

#denoise = ReguIsoNonlinear(left, 100, 10, sigma=0.5)
#f=plt.figure(figsize=(10,8))
#plt.imshow(denoise, cmap=plt.cm.gray)
#plt.show()
#
#denoise_2 = EdgeEnhance2D(left, 100, 10, sigma=0.5)
#f=plt.figure(figsize=(10,8))
#plt.imshow(denoise_2, cmap=plt.cm.gray)
#plt.show()

# In[]
# Tip image analysis for Xiaosh

#part = image
#hori = np.sum(part, axis=0)/y_lim
#thres = GaussianMixThres(hori, components=2, scale=0.5)
#hori = AnisotropicFilter1D(hori, 10, 2, delta_t=0.3)
#plt.plot(hori)
#plt.plot([0, len(hori)-1], [thres, thres])                                       
#plt.show()



#bi_line = hori < thres
#edges = np.diff(bi_line.astype(int))
#up = np.argwhere(edges == 1).flatten()
#down = np.argwhere(edges == -1).flatten()
#up_idx = np.argmin(np.abs(up - x_lim/2))
#down_idx = np.argmin(np.abs(down-x_lim/2))
#sep_line = int((up[up_idx] + down[down_idx])/2)
#

#scan_avg = 2
#
#left = np.sum(part[:,:sep_line], axis=1)/sep_line
#right = np.sum(part[:,sep_line:], axis=1)/sep_line
#
#left = AnisotropicFilter1D(left, 50, 10, delta_t=0.3)
#left_count, left_y, _ = OneDChannel(left, target='dark')
#plt.plot(left)
#for i in range(left_count):
#    plt.plot([left_y[i], left_y[i]], [np.max(left), np.min(left)])
#plt.show()
#
#left_x = []
#for y in left_y:
#    y = int(y)
#    seg = np.sum(part[y-scan_avg:y+scan_avg, :sep_line], axis=0)/(2*scan_avg)
#    seg = AnisotropicFilter1D(seg, 50, 50, delta_t=0.3)
#    left_x.append(np.argmax(seg))
#    
#right = AnisotropicFilter1D(right, 50, 10, delta_t=0.3)
#right_count, right_y, _ = OneDChannel(right, target='dark')
#plt.plot(right)
#for i in range(right_count):
#    plt.plot([right_y[i], right_y[i]], [np.max(right), np.min(right)])
#plt.show()
#
#right_x = []
#for y in right_y:
#    y = int(y)
#    seg = np.sum(part[y-scan_avg:y+scan_avg, sep_line:], axis=0)/(2*scan_avg)
#    seg = AnisotropicFilter1D(seg, 50, 50, delta_t=0.3)
#    right_x.append(np.argmax(seg)+sep_line)
#
#f = plt.figure(figsize=(8,8))
#plt.imshow(part, cmap=plt.cm.gray)
#plt.plot([sep_line, sep_line], [0, y_lim], 'r', linestyle='--')
#for i, j in zip(zip(left_x, right_x), zip(left_y, right_y)):
#    plt.plot(i, j, 'b')
#plt.ylim([y_lim, 0])
#plt.show()
#plt.savefig('tips.png',dpi=800)