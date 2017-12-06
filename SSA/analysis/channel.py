# -*- coding: utf-8 -*-
"""
This scrip contains all the front end algorithm I developed for channel-like images.
Most of them are for specific plugins. They are under constant development
"""
import numpy as np
import scipy.signal as spysig
from . import GeneralProcess
import matplotlib.pyplot as plt
from skimage import exposure

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
    # Sort peaks based on the intensity of the gradient
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

def VertiChannel(image, reference, scale=0.5, mode='up', target='dark', quality='uneven'):
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
    
    kappa = 0.03 * (np.max(hori) - np.min(hori))
    hori = GeneralProcess.ReguIsoNonlinear(hori, 50, kappa,
                                   sigma=0.5, delta_t=0.1)
    
    hori_cdf, hori_cdf_centers = exposure.cumulative_distribution(hori)
        
    if quality == 'equ_dist':
        peak_value = hori_cdf_centers[np.argmin(np.abs(hori_cdf - 0.5))]
        means_init=[[0.5 * (peak_value+np.min(hori_cdf_centers))], [peak_value]]
        thres = GeneralProcess.GaussianMixThres(hori, means_init=means_init, 
                                                components=2, scale=scale)
    if quality == 'uneven' or thres is None:
        if mode == 'up':
            seg = image[int(round(reference)):, :]
        elif mode == 'down':
            seg = image[:int(round(reference)), :]
            
        hist, bin_centers = exposure.histogram(seg)
        plt.plot(bin_centers, hist)
        plt.show()

        img_cdf, cdf_centers = exposure.cumulative_distribution(seg)          
        try:
            cutoff = cdf_centers[np.argmin(np.abs(img_cdf - 0.85))]
            data = seg.flatten()
            data = np.ma.masked_where(data>=cutoff, data)
            data = data[~data.mask]
            data_cdf, data_cdf_center = exposure.cumulative_distribution(data)
            hist, bin_centers = exposure.histogram(data)
            plt.plot(bin_centers, hist)
            plt.show()
            
            mu1 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.10))]
            sigma1 = mu1 - data_cdf_center[np.argmin(np.abs(data_cdf - 0.04))]
            mu2 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.5))]
            sigma2 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.8))] - mu2            
            separation = data_cdf_center[np.argmin(np.abs(data_cdf - 0.35))]
            bounds = ([0, np.min(data_cdf_center), 0, separation, 0], 
                       [np.inf, separation, np.inf, np.max(data_cdf_center), np.inf])
            thres = GeneralProcess.BiGaussCDFThres(data_cdf, data_cdf_center, 
                                                      init=[0.2, mu1, sigma1, mu2, sigma2],
                                                      bounds=bounds, plot=True)
        except Exception as e:
            print(str(e))
     
    plt.plot(hori)
    plt.plot([0, len(hori)-1], [thres, thres])                                       
    plt.show()  

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
            
            if len(up) < 10 :
                plateau = np.insert(plateau, 0, 0)
                plateau = np.append(plateau, x_lim-1)
            else:
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


def BimodalThres(image, plot=False, parameters=False):
    
    img_cdf, cdf_center = exposure.cumulative_distribution(image)
    cutoff = cdf_center[np.argmin(np.abs(img_cdf - 0.85))]
    data = image.flatten()
    data = np.ma.masked_where(data>=cutoff, data)
    data = data[~data.mask]
    data_cdf, data_cdf_center = exposure.cumulative_distribution(data)
    if plot:
        hist, bin_centers = exposure.histogram(data)
        plt.plot(bin_centers, hist)
        plt.show()
    
    mu1 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.10))]
    sigma1 = mu1 - data_cdf_center[np.argmin(np.abs(data_cdf - 0.04))]
    mu2 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.5))]
    sigma2 = data_cdf_center[np.argmin(np.abs(data_cdf - 0.8))] - mu2            
    separation = data_cdf_center[np.argmin(np.abs(data_cdf - 0.35))]
    bounds = ([0, np.min(data_cdf_center), 0, separation, 0], 
               [np.inf, separation, np.inf, np.max(data_cdf_center), np.inf])
    results = GeneralProcess.BiGaussCDFThres(data_cdf, data_cdf_center, 
                                              init=[0.2, mu1, sigma1, mu2, sigma2],
                                              bounds=bounds, plot=plot, 
                                              parameters=parameters)
    return results

def crossing(profile, threshold):
    bi_line = profile < threshold
    edges = np.diff(bi_line.astype(int))
    up = np.argwhere(edges == 1).flatten()
    down = np.argwhere(edges == -1).flatten()
    return up, down

def HalfSplit(image, mode='Vertical', plot=False):
    y_lim, x_lim = image.shape
        
    if mode == 'Vertical':
        hori = np.sum(image, axis=0) / y_lim
    elif mode == 'Horizontal': 
        hori = np.sum(image, axis=1) / x_lim
    
    thres, popt = BimodalThres(image, plot=plot, parameters=True)
    mu2 = popt[3]
    sigma2 = popt[4]
    limit = mu2 + sigma2 * 0.2
    
    up, down = crossing(hori, thres)
    up_idx = np.argmin(np.abs(up - x_lim/2))
    down_idx = np.argmin(np.abs(down - x_lim/2))
    split = int((up[up_idx] + down[down_idx])/2)
    
    up, down = crossing(hori, limit)
    left = np.min(up)
    right = np.max(down)
    
    if plot:
        plt.plot(hori)
        plt.plot([0, len(hori)-1], [thres, thres])
        plt.plot([0, len(hori)-1], [mu2, mu2], label='mu')
        plt.plot([0, len(hori)-1], [limit, limit], label='limit')
        plt.legend(loc=2)
        plt.show() 
    return split, left, right

def ChannelCD(image, measured_lvl, reference, algo='fit', 
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
#    if find_ref is None:
#        reference = 0
#        mode = 'up'
#    elif find_ref is True:
#        reference = IntensInterface(image, ref_range=ref_range)
#    else:
#        try:
#            reference = int(round((find_ref[0][1] + find_ref[1][1])/2))
#        except ValueError:
#            print('The format of reference line is incorrect')
    
    channel_count, channel_center, plateau = VertiChannel(image, reference, mode=mode, scale=0.4)
    
    channel_cd = [[] for _ in range(channel_count)]
    channel_points = [[] for _ in range(channel_count)]
    if mode == 'up':
        # Mode up means the reference line is higher
        measured_lvl = measured_lvl + int(round(reference))
    elif mode == 'down':
        # Mode down means the reference line is lower
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
    
    return channel_count, channel_cd, channel_points, channel_center, plateau


def BulkCD(image, sample, start_from, algo='fit', find_ref=True, ref_range=[0,np.Inf], 
              scan=1, threshold=100, mode='up'):
    '''
    To measure CD of three bulk region of Y-cut 
    '''
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
    
    y_lim, x_lim = image.shape
    if mode == 'mask' or mode == 'down':
        channel_count, channel_center, plateau = VertiChannel(image, reference, 
                                                              mode='down', scale=0.2, 
                                                              target='dark', quality='uneven')
        lvls = np.arange(10, reference - start_from, sample)
            
    elif mode == 'up':
        channel_count, channel_center, plateau = VertiChannel(image, reference, 
                                                              mode='up', scale=0.2, 
                                                              target='dark', quality='uneven')
        lvls = np.arange(reference + start_from, y_lim, sample)
    
    channel_width = np.diff(channel_center)
    channel_width = list(zip(channel_width, range(len(channel_width))))
    channel_width.sort(reverse=True)
    bulk_idx = np.array(channel_width)[:3, 1].astype(int)
        
    channel_cd = [[] for _ in range(len(bulk_idx) * 2)]
    channel_points = [[] for _ in range(len(bulk_idx) * 2)]
    
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
    

    for lvl in lvls:
        line_seg = np.sum(image[lvl-before:lvl+after,:], axis=0) / scan
        for i in range(len(bulk_idx)):
            idx = bulk_idx[i]         
            left_center = int(round(channel_center[idx]))
            left_left = int(round(plateau[idx]))
            left_right = int(round(plateau[idx + 1]))
            x_L = np.arange(left_left, left_center)
            y_L = line_seg[left_left:left_center]
            x_R = np.arange(left_center, left_right)
            y_R = line_seg[left_center:left_right]
                
            if algo == 'fit':
                left_left_edge = GeneralProcess.LGSemEdge(x_L, y_L, threshold=threshold, 
                                                 orientation='backward')                   
                left_right_edge = GeneralProcess.LGSemEdge(x_R, y_R, threshold=threshold,
                                                  orientation='forward')
                if left_left_edge is None or left_right_edge is None:
                    left_left_edge, left_right_edge = ChannelEdgeIntens(line_seg[left_left:left_right], 
                                                              left_center-left_left, vert_height=threshold/100.0)
                    left_left_edge += left_left
                    left_right_edge += left_left
            
            channel_cd[2*i].append(left_right_edge-left_left_edge)
            channel_points[2*i].append([[left_left_edge, lvl], [left_right_edge, lvl]])
                
            right_center = int(round(channel_center[idx + 1]))
            right_left = int(round(plateau[idx + 1]))
            right_right = int(round(plateau[idx + 2]))
            x_L = np.arange(right_left, right_center)
            y_L = line_seg[right_left:right_center]
            x_R = np.arange(right_center, right_right)
            y_R = line_seg[right_center:right_right]
            
            if algo == 'fit':
                right_left_edge = GeneralProcess.LGSemEdge(x_L, y_L, threshold=threshold, 
                                                 orientation='backward')                   
                right_right_edge = GeneralProcess.LGSemEdge(x_R, y_R, threshold=threshold,
                                                  orientation='forward')
                if right_left_edge is None or right_right_edge is None:

                    right_left_edge, right_right_edge = ChannelEdgeIntens(line_seg[right_left:right_right], 
                                                              right_center-right_left, vert_height=threshold/100.0)
                    right_left_edge += right_left
                    right_right_edge += right_left

            channel_cd[2*i+1].append(right_right_edge-right_left_edge)
            channel_points[2*i+1].append([[right_left_edge, lvl], [right_right_edge, lvl]])
                
    return 6, reference, channel_cd, lvl, channel_points, channel_center, plateau
    
    

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
        vert_seg = np.sum(image[ref:,channel_loc:channel_loc+scan], axis=1) / scan     
        vert_seg = GeneralProcess.AnisotropicFilter1D(vert_seg, iteration, noise, delta_t=0.3)
        max_depth = len(vert_seg)
        # Try to find the bottom starting from the middle
        x = np.arange(int(max_depth/2), max_depth)
        y = vert_seg[int(max_depth/2):]
        if mag == 'high':
            bot = GeneralProcess.LGSemEdge(x, y, threshold=threshold,
                                                  orientation='forward')
            print(bot)
            if bot is None:
                bot = np.argmax(vert_seg[int(max_depth/2):]) + max_depth/2
        elif mag == 'low':
            bot = np.argmax(vert_seg[int(max_depth/2):]) + max_depth/2
        bot += ref
        depth.append(bot-ref)
        depth_points.append([[channel_loc, ref], [channel_loc, bot]])   
    return channel_count, depth, depth_points, channel_center, plateau

def AEPCSpecial(image, ref, scan, threshold=100, noise=5000, iteration=0, mode='up',
         mag='high', algo='SEM', bot_diff=0):
    y_lim, x_lim = image.shape
    channel_count, channel_center, plateau = VertiChannel(image, ref, mode=mode)
    depth = []
    depth_points = [] 
    channel_cd = [[] for _ in range(channel_count)]
    channel_points = [[] for _ in range(channel_count)]
    
    if scan % 2 == 0:
        before = scan // 2
        after = scan //2
    else:
        before = scan // 2
        after = scan // 2 + 1
    
    for i in range(channel_count):
        channel_loc = int(round(channel_center[i]))
        vert_seg = np.sum(image[ref:,channel_loc-before:channel_loc+after], axis=1) / scan        
        vert_seg = GeneralProcess.AnisotropicFilter1D(vert_seg, iteration, noise, delta_t=0.3)
        max_depth = len(vert_seg)
        x = np.arange(int(max_depth/2), max_depth)
        y = vert_seg[int(max_depth/2):]
        if mag == 'high':
            if algo == 'STEM':
                bot = GeneralProcess.SigmoEdge(x, y, threshold=threshold,
                                                  orientation='forward')
            elif algo == 'SEM':
                bot = GeneralProcess.LGSemEdge(x, y, threshold=threshold,
                                                  orientation='forward')
            if bot is None:
                bot = np.argmax(vert_seg[int(max_depth/2):]) + max_depth/2
        elif mag == 'low':
            bot = np.argmax(vert_seg[int(max_depth/2):]) + max_depth/2
        bot += ref       
        depth.append(bot-ref)
        depth_points.append([[channel_loc, ref], [channel_loc, bot]])
    
    avg_depth = np.average(depth)
    measured_lvl = [ref, int(round((avg_depth*0.5))) + ref , ref + int(round(avg_depth - bot_diff))]

    for lvl in measured_lvl:
        hori_seg = np.sum(image[lvl-before:lvl+after,:], axis=0) / scan
        hori_seg = GeneralProcess.AnisotropicFilter1D(hori_seg, iteration, noise, delta_t=0.3)
        for i in range(channel_count):
            center = int(round(channel_center[i]))
            left = int(round(plateau[i]))
            right = int(round(plateau[i+1]))
            x_L = np.arange(left, center)
            y_L = hori_seg[left:center]
            x_R = np.arange(center, right)
            y_R = hori_seg[center:right]           

            left_edge = GeneralProcess.LGSemEdge(x_L, y_L, threshold=threshold, 
                                             orientation='backward')                   
            right_edge = GeneralProcess.LGSemEdge(x_R, y_R, threshold=threshold,
                                              orientation='forward')
            if left_edge is None or right_edge is None:
                left_edge, right_edge = ChannelEdgeIntens(hori_seg[left:right], 
                                                          center-left, vert_height=threshold/100.0)
                left_edge += left
                right_edge += left
            channel_cd[i].append(right_edge-left_edge)
            channel_points[i].append([[left_edge, lvl], [right_edge, lvl]])
   
    for i in range(channel_count):
        channel_cd[i].append(depth[i])
        channel_points[i].append(depth_points[i])
    line_modes = ['Horizontal', 'Horizontal', 'Horizontal', 'Vertical']
    return channel_count, channel_cd, channel_points, channel_center, plateau, line_modes


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
        # Try to find the bottom starting from the middle
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


def FourierTilt(image, ref, tot_num_channels, start=0, end=np.infty, scan=1):
    y_lim, x_lim = image.shape
    guess_freq = tot_num_channels / float(x_lim)
    upper_lim = guess_freq * 1.5
    lower_lim = guess_freq * 0.5
    row = np.arange(ref + start, min(end + ref, y_lim), scan)      
    theta = []
    idx_list = []
    for i in row:
        hori = np.sum(image[i:i+scan,:], axis=0)
        ft = np.fft.fft(hori) / 10**7
        freq = np.fft.fftfreq(hori.shape[-1])        
        posi_mask = np.logical_and(freq > lower_lim, freq < upper_lim) 
        freq = freq[posi_mask]
        ft = ft[posi_mask]
        idx = np.argmax(np.abs(ft))
        idx_list.append(idx)
        
    avg_idx = int(np.mean(idx_list))
    avg_freq = freq[avg_idx]
        
    for i in row:
        hori = np.sum(image[i:i+scan,:], axis=0)
        ft = np.fft.fft(hori) / 10**7
        freq = np.fft.fftfreq(hori.shape[-1])
        posi_mask = np.logical_and(freq > lower_lim, freq < upper_lim)
        ft = ft[posi_mask]
        angle = np.arctan2(ft.imag[avg_idx], ft.real[avg_idx])
        if len(theta) > 0:
            if angle - theta[-1] > np.pi:
                angle -= np.pi * 2
            elif angle - theta[-1] < - np.pi:
                angle += np.pi * 2
        theta.append(angle)
    
    
    theta = - (theta - theta[0])
    pix_shift = (theta / (2 * np.pi)) / avg_freq
    row = row - ref    
    theta = np.array(theta)
   
    z_ref = np.polyfit(row, pix_shift, 1)
    p_ref = np.poly1d(z_ref)
    f = plt.figure(figsize=plt.figaspect(3))
    plt.plot((-theta*180/np.pi), row, '.')
    plt.xlabel('Phase')
    plt.show()
    print('The angle is: %.3f'%(np.arctan(z_ref[0]) * 180/np.pi))
    plt.plot(row, pix_shift, '.')
    plt.plot(row, p_ref(row), 'r')
    plt.xlabel('Pixel')
    plt.show()
    
    return row, pix_shift


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
        # mode up means the reference line is higher
        measured_lvl = measured_lvl + int(round(reference))
    elif mode == 'down':
        # mode down means the reference line is lower
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