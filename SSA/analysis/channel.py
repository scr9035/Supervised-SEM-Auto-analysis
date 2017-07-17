# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

import numpy as np
import scipy.signal as spysig
from . import GeneralProcess
import matplotlib.pyplot as plt


def IntensInterface(image, ref_range=[0,np.Inf], axis=1, smooth=2000, iteration=10, 
                    width=np.arange(5,10)):
    """
    ref_range: the range to find the interface
    """
    y_lim, x_lim = image.shape
    if ref_range is not None:
        try:
            low_lim, high_lim = ref_range
            low_lim = max(0, low_lim)
            if axis == 1:
                high_lim = min(high_lim, y_lim)
            elif axis == 0:
                high_lim = min(high_lim, x_lim)
        except:
            low_lim = 0
            if axis == 1:
                high_lim = y_lim
            elif axis == 0:
                high_lim = x_lim
    sum_size = abs(high_lim - low_lim)
    
    if axis == 1:
        one_d_avg = np.sum(image[low_lim:high_lim,:], axis=axis)/float(sum_size)
    elif axis == 0:
        one_d_avg = np.sum(image[:,low_lim:high_lim], axis=axis)/float(sum_size)
        
    grad = np.gradient(one_d_avg)
    grad = GeneralProcess.AnisotropicFilter1D(grad, iteration, smooth, delta_t=0.3)
    peak_pos = spysig.find_peaks_cwt(grad, width)
    candidates = list(zip(grad[peak_pos], peak_pos))
    # DYL: sort peaks based on the intensity of the gradient
    candidates.sort()
    candidates.reverse()
    return int(round(np.array(candidates)[0,1])) + low_lim

def ChannelEdgeIntens(line, center, vert_height=1):
    left_peak = np.argmax(line[:center])
    right_peak = np.argmax(line[center:]) + center
    
    if vert_height == 1:
        return left_peak, right_peak
    else:
        left_height = line[left_peak] * vert_height
        right_height = line[right_peak] * vert_height 
        for i in reversed(np.arange(left_peak,center)):
            if line[i] >= left_height:
                left_edge = i
                break
        for i in np.arange(center, right_peak+1):
            if line[i] >= right_height:
                right_edge = i 
                break
        return left_edge, right_edge

def ChannelEdgeGrad(line, center, vert_height=1):
    grad = np.gradient(line.astype(float))
    left_peak = np.argmin(grad)
    right_peak = np.argmax(grad)
    left_grad = grad[left_peak] * vert_height
    right_grad = grad[right_peak] * vert_height
   
    for i in reversed(np.arange(0, left_peak+1)):
        if grad[i] >= left_grad:
            left_edge = i
            break
    for i in np.arange(right_peak, len(grad)):
        if grad[i] <= right_grad:
            right_edge = i
            break
    return left_edge, right_edge

def VertiChannel(image, reference, scale=0.5, mode='up', target='dark'):
    '''
    mode : str
        up, if the reference line is located higher than the channels
        down: if the reference line is located lower than the cd lines
    target : str, 'dark' or 'bright'
    '''
    y_lim, x_lim = image.shape
    if mode == 'up':
        hori = np.sum(image[int(round(reference)):, :], axis=0)/(y_lim-int(round(reference)))
    elif mode == 'down':
        hori = np.sum(image[:int(round(reference)), :], axis=0)/(int(round(reference)))
    thres = GeneralProcess.GaussianMixThres(hori, components=2, scale=scale)
#    plt.plot(hori)
#    plt.plot([0, len(hori)-1], [thres, thres])                                       
#    plt.show()  
    if target == 'dark':
        bi_line = hori > thres
    elif target == 'bright':
        bi_line = hori < thres
    else:
        return None, None, None
    edges = np.diff(bi_line.astype(int))
    up = np.argwhere(edges == 1).flatten()
    down = np.argwhere(edges == -1).flatten()
    
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


def ChannelCD(image, measured_lvl, algo='fit', find_ref=True, ref_range=[0,np.Inf], 
              scan=1, threshold=100, noise=5000, iteration=0, mode='up'):
    """
    Find CD lines and CDs for multiple vertial straight channels.
    
    Find reference line first, if it's not provided. 
    Get CD lines at the selected levels. Edge is selected based on the vertical 
    amplitude.
    
    Parameters:
    -----------
    
    find_ref : optional, boolean, np.array, or None. Default True
        Whether to find the reference line automatically.
    
    ref_range : list
        If is asked to find reference line automatically, this is the range to look
        for
    
    mode : str, optional, 'up' or 'down'
        'up' means reference line is higher than CD lines.
        'down' means reference lien is lower than CD lines
    
    """
    if find_ref is None:
        reference = 0
        mode = 'up'
    elif find_ref is True:
        reference = IntensInterface(image, ref_range=ref_range)
    else:
        try:
            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
        except ValueError:
            print('The format of reference line is incorrect')
    
    channel_count, channel_center, plateau = VertiChannel(image, reference, mode=mode)
    
    channel_cd = [[] for _ in range(channel_count)]
    channel_points = [[] for _ in range(channel_count)]
    if mode == 'up':
        # DYL: mode up means the reference line is higher
        measured_lvl = measured_lvl + int(round(reference))
    elif mode == 'down':
        # DYL: mode down means the reference line is lower
        measured_lvl = int(round(reference)) - measured_lvl
                          
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
        
    for lvl in measured_lvl:
        line_seg = np.sum(image[lvl-before:lvl+after,:], axis=0) / scan
#        line_seg = GeneralProcess.AnisotropicFilter1D(line_seg, iteration, noise, delta_t=0.3)
        for i in range(channel_count):
            center = int(round(channel_center[i]))
            left = int(round(plateau[i]))
            right = int(round(plateau[i+1]))
            x_L = np.arange(left, center)
            y_L = line_seg[left:center]
            x_R = np.arange(center, right)
            y_R = line_seg[center:right]
            
            if algo == 'fit':
                left_edge = GeneralProcess.LGSemEdge(x_L, y_L, threshold=threshold, 
                                                 orientation='backward')                   
                right_edge = GeneralProcess.LGSemEdge(x_R, y_R, threshold=threshold,
                                                  orientation='forward')
                if left_edge is None or right_edge is None:
                    left_edge, right_edge = ChannelEdgeIntens(line_seg[left:right], 
                                                              center-left, vert_height=threshold/100.0)
                    left_edge += left
                    right_edge += left
            elif algo == 'easy':
                left_edge, right_edge = ChannelEdgeIntens(line_seg[left:right], 
                                                              center-left, vert_height=threshold/100.0)
                left_edge += left
                right_edge += left
            channel_cd[i].append(right_edge-left_edge)
            channel_points[i].append([[left_edge, lvl], [right_edge, lvl]])
#            channel_points[i].append([(left_edge, lvl), (right_edge, lvl)])
    
    return channel_count, reference, channel_cd, channel_points, channel_center, plateau

def ChannelDepth(image, ref, scan=1, threshold=100, noise=5000, iteration=0, 
                 mode='up', mag='high'):
    """
    Find CD lines and CDs for multiple vertial straight channels.
    
    Find reference line first, if it's not provided. 
    Get CD lines at the selected levels. Edge is selected based on the vertical 
    amplitude.
    """
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, ref, mode=mode)
    depth = []
    depth_points = []    
    for i in range(channel_count):
        channel_loc = int(round(channel_center[i]))
        line_seg = np.sum(image[ref:,channel_loc:channel_loc+scan], axis=1) / scan        
        line_seg = GeneralProcess.AnisotropicFilter1D(line_seg, iteration, noise, delta_t=0.3)
        max_depth = len(line_seg)
        # DYL: try to find the bottom starting from the middle
        x = np.arange(int(max_depth/2), max_depth)
        y = line_seg[int(max_depth/2):]
        if mag == 'high':
            bot = GeneralProcess.LGSemEdge(x, y, threshold=threshold,
                                                  orientation='forward')
            if bot is None:
                bot = np.argmax(line_seg[int(max_depth/2):]) + max_depth/2
        elif mag == 'low':
            bot = np.argmax(line_seg[int(max_depth/2):]) + max_depth/2
        bot += ref
        depth.append(bot-ref)
        depth_points.append([[channel_loc, ref], [channel_loc, bot]])   
    return channel_count, depth, depth_points, channel_center, plateau

def RecessDepth(image, ref, scan=1, threshold=100, mode='up'):
    """
    Find CD lines and CDs for multiple vertial straight channels.
    
    Find reference line first, if it's not provided. 
    Get CD lines at the selected levels. Edge is selected based on the vertical 
    amplitude.
    """
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, ref, mode=mode)
    depth = []
    depth_points = []    
    for i in range(channel_count):
        channel_loc = int(channel_center[i])
        line_seg = np.sum(image[ref:,channel_loc:channel_loc+scan], axis=1) / scan        
        # DYL: try to find the bottom starting from the middle
        x = np.arange(len(line_seg))
        y = line_seg
        bot = None # Bypass the LGSemEdge algorithm. Use the simple one
#        bot = GeneralProcess.LGSemEdge(x, y, threshold=threshold,
#                                              orientation='forward')
        if bot is None:
            bot = np.argmax(line_seg)
        bot += ref
        depth.append(bot-ref)
        depth_points.append([[channel_loc, ref], [channel_loc, bot]])   
    return channel_count, depth, depth_points, channel_center, plateau

def RemainMask(image, ref, scan=1, noise=5000, iteration=0):
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, ref)
    height = []
    top_points = []
    for i in range(channel_count):
        plateau_loc = int(round(plateau[i]))
        line_seg = np.sum(image[:ref, plateau_loc:plateau_loc+scan], axis=1) / scan        
        line_seg = GeneralProcess.AnisotropicFilter1D(line_seg, iteration, noise, delta_t=0.3)
        top = np.argmax(line_seg)
        height.append(ref-top)
        top_points.append([[plateau_loc, top], [plateau_loc, ref]])   
    return channel_count, height, top_points, channel_center, plateau


def FinEdge(image, measured_lvl, find_ref=True, threshold=99, ref_range=[0,np.Inf], 
            scan=1, mode='up', field='dark'):
    if find_ref is None:
        reference = 0
        mode = 'up'
    elif find_ref is True:
        reference = IntensInterface(image, ref_range=ref_range)
    else:
        try:
            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
        except ValueError:
            print('The format of reference line is incorrect')
    
    if field == 'dark':
        target = 'bright'
    elif field == 'bright':
        target = 'dark'
    
    channel_count, channel_center, plateau = VertiChannel(image, reference, 
                                                          mode=mode, target=target)
    channel_cd = [[] for _ in range(channel_count)]
    channel_points = [[] for _ in range(channel_count)]
    
    if mode == 'up':
        # DYL: mode up means the reference line is higher
        measured_lvl = measured_lvl + int(round(reference))
    elif mode == 'down':
        # DYL: mode down means the reference line is lower
        measured_lvl = int(round(reference)) - measured_lvl
    
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
    
    for lvl in measured_lvl:
        for i in range(channel_count):
            line_seg = np.sum(image[lvl-before:lvl+after,:], axis=0) / scan
                             
            center = int(round(channel_center[i]))
            left = int(round(plateau[i]))
            right = int(round(plateau[i+1]))
    
            x1 = np.arange(left, center)
            y1 = line_seg[left:center]
            y1 = np.array(y1, dtype='int64')
            
            x2 = np.arange(center, right)
            y2 = line_seg[center:right]
            y2 = np.array(y2, dtype='int64')
            
            if field == 'bright':
                left_edge = GeneralProcess.SigmoEdge(x1, y1, threshold=threshold, 
                                                     orientation='backward')
                right_edge = GeneralProcess.SigmoEdge(x2, y2, threshold=threshold, 
                                                      orientation='forward')
            elif field == 'dark':
                left_edge = GeneralProcess.SigmoEdge(x1, y1, threshold=threshold, 
                                                     orientation='forward')
                right_edge = GeneralProcess.SigmoEdge(x2, y2, threshold=threshold, 
                                                      orientation='backward')
#            left_edge = GeneralProcess.LGStemEdge(x1, y1, threshold=threshold, 
#                                                 orientation='backward')               
#            right_edge = GeneralProcess.LGStemEdge(x2, y2, threshold=threshold, 
#                                                  orientation='forward')          
            channel_cd[i].append(right_edge-left_edge)
            channel_points[i].append([[left_edge, lvl], [right_edge, lvl]]) 
    return channel_count, reference, channel_cd, channel_points, channel_center, plateau


def FinDepth(image, find_ref=True, threshold=99, ref_range=[0,np.Inf], 
            scan=1, mode='up', field='dark'):
    if find_ref is None:
        reference = 0
        mode = 'up'
    elif find_ref is True:
        reference = IntensInterface(image, ref_range=ref_range)
    else:
        try:
            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
        except ValueError:
            print('The format of reference line is incorrect')
    
    if field == 'dark':
        target = 'dark'
        ori = 'forward'
    elif field == 'bright':
        target = 'bright'
        ori = 'backward'
        
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, reference, 
                                                          mode=mode, target=target)
    
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
        
    depth = []
    depth_points = []
    
    for i in range(channel_count):
        channel_loc = int(round(channel_center[i]))
        intens = np.sum(image[reference:,channel_loc-before:channel_loc+after], axis=1) / scan
        pos = np.arange(reference, y_lim)
        bot = GeneralProcess.SigmoEdge(pos, intens, threshold=threshold, orientation=ori)
        depth.append(bot-reference)
        depth_points.append([[channel_loc, reference], [channel_loc, bot]]) 
            
    return channel_count, reference, depth, depth_points, channel_center, plateau

def FinRHM(image, find_ref=True, threshold=99, ref_range=[0,np.Inf], 
            scan=1, mode='up', field='dark'):
    if find_ref is None:
        reference = 0
        mode = 'up'
    elif find_ref is True:
        reference = IntensInterface(image, ref_range=ref_range)
    else:
        try:
            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
        except ValueError:
            print('The format of reference line is incorrect')
    
    if field == 'dark':
        target = 'bright'
        ori = 'forward'
    elif field == 'bright':
        target = 'dark'
        ori = 'backward'
        
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, reference, 
                                                          mode=mode, target=target)
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
        
    RHM = []
    mask_points = []
    
    for i in range(channel_count):
        channel_loc = int(round(channel_center[i]))
        intens = np.sum(image[:reference,channel_loc-before:channel_loc+after], axis=1) / scan
        pos = np.arange(0, reference)
        top = GeneralProcess.SigmoEdge(pos, intens, threshold=threshold, orientation=ori)
        RHM.append(reference - top)
        mask_points.append([[channel_loc, top], [channel_loc, reference]])
            
    return channel_count, reference, RHM, mask_points, channel_center, plateau


def StemEdge(image, measured_lvl, find_ref=True, threshold=99, ref_range=[0,np.Inf], 
             mode='up'):
    if find_ref is None:
        reference = 0
        mode = 'up'
    elif find_ref is True:
        reference = IntensInterface(image, ref_range=ref_range)
    else:
        try:
            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
        except ValueError:
            print('The format of reference line is incorrect')
            
    channel_count, channel_center, plateau = VertiChannel(image, 0, 
                                                          mode=mode, target='dark')
    channel_cd = [[] for _ in range(channel_count)]
    channel_points = [[] for _ in range(channel_count)]
    
    if mode == 'up':
        # DYL: mode up means the reference line is higher
        measured_lvl = measured_lvl + int(round(reference))
    elif mode == 'down':
        # DYL: mode down means the reference line is lower
        measured_lvl = int(round(reference)) - measured_lvl
                          
    for lvl in measured_lvl:
        for i in range(channel_count):
            center = int(round(channel_center[i]))
            left = int(round(plateau[i]))
            right = int(round(plateau[i+1]))
    
            x1 = np.arange(left, center)
            y1 = image[lvl, left:center]
            y1 = np.array(y1, dtype='int64')
            left_edge = GeneralProcess.SigmoEdge(x1, y1, threshold=threshold, 
                                                 orientation='backward')
#            left_edge = GeneralProcess.LGStemEdge(x1, y1, threshold=threshold, 
#                                                 orientation='backward')          
            x2 = np.arange(center, right)
            y2 = image[lvl, center:right]
            y2 = np.array(y2, dtype='int64')
            
            right_edge = GeneralProcess.SigmoEdge(x2, y2, threshold=threshold, 
                                                  orientation='forward')
#            right_edge = GeneralProcess.LGStemEdge(x2, y2, threshold=threshold, 
#                                                  orientation='forward')          
            channel_cd[i].append(right_edge-left_edge)
            channel_points[i].append([[left_edge, lvl], [right_edge, lvl]])            
    return channel_count, reference, channel_cd, channel_points, channel_center, plateau


def StemDepth(image, ref, threshold=20, scan=1, mode='up', target='bright'):
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, 0, mode=mode, target=target)
    depth = []
    depth_points = []    
    for i in range(channel_count):
        channel_loc = int(round(channel_center[i]))
        intens = np.sum(image[ref:,channel_loc:channel_loc+scan], axis=1) / scan
        pos = np.arange(ref, y_lim)
        bot = GeneralProcess.SigmoEdge(pos, intens, threshold=threshold, orientation='backward')
#        bot = GeneralProcess.LGStemEdge(pos, intens, threshold=threshold, 
#                                    orientation='backward', mode='STEM')
        depth.append(bot-ref)
        depth_points.append([[channel_loc, ref], [channel_loc, bot]])   
    return channel_count, depth, depth_points, channel_center, plateau