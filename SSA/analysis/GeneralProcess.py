# -*- coding: utf-8 -*-
#
"""
This script mostly contains general algorithms I developed for image analysis.
Not all of them are useful and proper. And they are under constant development.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skimorph
import skimage.filters as skifilter
import sklearn.neighbors as skneighbor 
import scipy.odr as spyodr
import sklearn.mixture as skMixture
from scipy.optimize import curve_fit
from skimage import exposure

def PixDistribution(image, bin_num=1000):
    values, bins = np.histogram(image, bins=bin_num)
    max_count = max(values)
    fig, ax = plt.subplots()
    plt.plot(bins[:-1], values, lw=2, c='k')
    return max_count, ax, fig

def GaussianMixThres(image, components=2, means_init=None, n_init=3, scale=0.5):
    gmix = skMixture.GaussianMixture(n_components=components, n_init=n_init,
                                     means_init=means_init,
                                     )
    hist, bin_centers = exposure.histogram(image)
    gmix.fit(image.flatten()[:, np.newaxis])
    mean1, mean2 = np.sort(gmix.means_[:,0])
    plt.plot(bin_centers, hist)
    plt.vlines(mean1, 0, 40)
    plt.vlines(mean2, 0, 40)
    plt.show()
    thres = mean1 + (mean2 - mean1) * scale
    if (mean2 - mean1) / thres < 0.2:
        return None
    else:
        return thres

def KernelThresh(image, intens=[0, 40000], num=4000, bandwidth=2000, 
                 kernel='gaussian'):
    """Determine threshold using Gaussian kernel density estimation
    
    This is good for bimodal distribution. Using Gaussian kernel density 
    estimation (KDE) to find the two mode of distribution. The threshold is 
    choosen as the middle of the two modes.
    """
    _max_count, _ax, _fig = PixDistribution(image)
    kde = skneighbor.KernelDensity(kernel=kernel, bandwidth=bandwidth)
    if len(image.shape) > 1:
        kde.fit(image.flatten()[:, np.newaxis])
    else:
        kde.fit(image[:, np.newaxis])
    x_pos = np.linspace(intens[0], intens[1], num=num)[:, np.newaxis]
    kde.get_params()
    log_dens = kde.score_samples(x_pos)
    dens = np.exp(log_dens)
    maxima = LocalMaxima(dens, width=100, highPeak=False)
    if len(maxima) != 2:
        print('Non-bimodal detected')
        return None
    else:
        m1, m2 = maxima
        thres = 0.5 * (x_pos[m1,0] + x_pos[m2,0])
        _ax.plot([thres, thres], [0, _max_count], label='Ostu')
        plt.show()
        return thres

def BinaryConverter(image, thres='otsu', scale=1, iteration=1):
    """Convert the image into a binary image.
    
    Parameters
    ----------
    thres : string or int, default 'otsu'
        If thres is 'otsu' then the threshold is choosen using the otsu method. If
        thres is a number then the number is used directly as threshold
    scale : float
        Scale the threshold by this number
    iterations : int
        Iteration for the binary opening operation. Larger number will remove the residue
    remove_label : Boolean
        Weathre to remove the SEM labels with scale bars and other information
        
    Returns
    -------
    bi_fig : (N, M) ndarray
        The binary image
    """
    if thres == 'otsu':
        thrs = skifilter.threshold_otsu(image, nbins=1024)
    else:
        thrs = thres
    bi_fig = image <= (thrs * scale)
#    bi_fig = ndimage.binary_opening(bi_fig, iterations=iteration)
#    bi_fig = ndimage.binary_closing(bi_fig)
    return bi_fig

def BinaryDialateEdge(bi_img):
    dialate_bi = skimorph.binary_dilation(bi_img, skimorph.diamond(1)).astype(np.uint8)
    edge = dialate_bi - bi_img
    edge = edge[1:-1,1:-1]
    return edge

def BinaryErosionEdge(bi_img):
    ero_bi = skimorph.binary_erosion(bi_img, skimorph.diamond(1)).astype(np.uint8)
    edge = ero_bi - bi_img
    edge = edge[1:-1,1:-1]
    return edge

def Intersection(line1, line2):
    """Compute the intersection of two 2D lines represted by two points on the line.
    """
    p_seg = line1[1] - line1[0]
    q_seg = line2[1] - line2[0]
    ini_seg = line1[0] - line2[1]
    coeffi = (q_seg[1] * ini_seg[0] - q_seg[0] * ini_seg[1]) / \
            float(q_seg[0] * p_seg[1] - q_seg[1] * p_seg[0])
    sect_point = line1[0] + coeffi * p_seg
    return sect_point

def ThreePointOrien(p1, p2, p3):
    # See http://www.geeksforgeeks.org/orientation-3-ordered-points/
    # for details of below formula.
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]);
    if (val == 0):
        return 0  # colinear 
    return 1 if val > 0 else 2 # clock or counterclock wise

def SegmentIntersect(line1, line2):
    p1, q1 = line1
    p2, q2 = line2
    if (ThreePointOrien(p1, q1, p2) != ThreePointOrien(p1, q1, q2)) and \
        (ThreePointOrien(p2, q2, p1) != ThreePointOrien(p2, q2, q1)):
        return True
    else:
        return False

def AnisotropicFilter1D(intensity, iteration, kappa, delta_t=0.2):
    finit_diff_mask = np.array([1, -1])
    de_noised = intensity.astype(float)
    while iteration > 0:
        iteration -= 1
        gradient = np.convolve(finit_diff_mask, de_noised, mode='valid')
        diffu_coef = np.exp(- gradient**2 / kappa**2)       
        flux = - delta_t * (diffu_coef * gradient)
        de_noised[:-1] = de_noised[:-1] - flux # left side flux
        de_noised[1:] = de_noised[1:] + flux # right side flux       
    return de_noised
    
def AnisotropicImageFilter1D(image, iteration, kappa, delta_t=0.3):
    """Reduce noise at one direction of an image
    
    """
    row, col = image.shape
    clean_image = np.zeros([row, col])
    for i in range(row):
        clean_image[i,:] = AnisotropicFilter1D(image[i,:], iteration, kappa, delta_t=delta_t)
    return clean_image

def AnisotropicFilter2D(image, iteration, kappa, delta_t=0.2):
    """Reduce noise of full image
    """
#    finite_diff_mask = np.array([1,-1])
    de_noised = image.astype(float)
    while iteration > 0:
        iteration -= 1
#        axisX_grad = ndimage.convolve1d(de_noised, finite_diff_mask, 
#                                        axis=1, mode='constant')[:,:-1]
#        axisY_grad = ndimage.convolve1d(de_noised, finite_diff_mask,
#                                        axis=0, mode='constant')[:-1,:]

        # The following is a faster way to calculate gradient than the convolution
        axisX_grad = de_noised[:,1:] - de_noised[:,:-1]        
        axisY_grad = de_noised[1:,:] - de_noised[:-1,:]        
        diffu_coef_X = np.exp(- axisX_grad**2 / kappa**2)
        diffu_coef_Y = np.exp(- axisY_grad**2 / kappa**2)
        flux_X = - delta_t * (diffu_coef_X * axisX_grad)
        flux_Y = - delta_t * (diffu_coef_Y * axisY_grad)
        de_noised[:, :-1] = de_noised[:, :-1] - flux_X
        de_noised[:, 1:] = de_noised[:, 1:] + flux_X
        de_noised[1:, :] = de_noised[1:,:] + flux_Y
        de_noised[:-1, :] = de_noised[:-1, :] - flux_Y
    return de_noised

def Logistic(x, x0, y0, k, c):
     y = c / (1 + np.exp(-k*(x - x0))) + y0
     return y

def LogisGradPercent(x0, k, percent, toward='high'):
    '''Return x coordinates with percentage of gradient in logistic
    
    toward : str, optional 'high' or 'low'
    '''
    p = 2 / np.sqrt(percent)
    if toward == 'high':
        return x0 - 2 * np.log((p - np.sqrt(p**2 - 4))/2) / k
    elif toward == 'low':
        return x0 - 2 * np.log((p + np.sqrt(p**2 - 4))/2) / k
    

def SigmoEdge(x, y, threshold=99, orientation='forward'):
    if orientation == 'forward':
        ori = 1
    elif orientation == 'backward':
        ori = -1
    mid = x[np.argmin(np.abs(y - (np.max(y)+np.min(y))/2))]

    p0 = [mid, np.min(y), ori, np.max(y)-np.min(y)]
    try:
        popt, pcov = curve_fit(Logistic, x, y, p0=p0)
        x0, y0, k, c = popt
        if threshold >= 0:
            edge = LogisGradPercent(x0, k, threshold/100.0, toward='high')
        else:
            edge = LogisGradPercent(x0, k, -threshold/100.0, toward='low')
#        edge = x0 - np.log(100.0/threshold-1) / k
    except RuntimeError:
        return x[int(len(x)/2)]      
#    plt.plot(x, Logistic(x, *popt))
#    plt.plot(x, y)
#    plt.plot([edge, edge], [np.min(y), np.max(y)])
#    plt.show()
    return edge

def LogiLorent(x, x0, y0, k, c, x_L, G):
    y = c / (1 + np.exp(-k*(x - x0))) + y0 + G**2 / (4 * (x - x_L)**2 + G**2)
    return y

def LogiGaussian(x, x0, y0, k, c, A, mu, sigma):
    y = c / (1 + np.exp(-k*(x - x0))) + y0 + A * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y

def LGSemEdge(x, y, threshold=100, finess=0.05, orientation='forward'):
    """
    finess :
        amount of pixel
    mode :
        SEM and STEM are different.
    """
    if orientation == 'forward':
        ori = 1
        y0 = y[0]
        c = y[-1]-y[0]
        A = np.max(y) - y[-1]
    elif orientation == 'backward':
        ori = -1
        y0 = y[-1]
        c = y[0]-y[-1]
        A = np.max(y) - y[0]

    mid = x[np.argmin(np.abs(y - (y[0] + y[-1])/2))]   
    mu = x[np.argmax(y)]
    sigma = np.abs(mu - mid)/2
    
    p0 = [mid, y0, ori, c, A, mu, sigma]
    try:
        popt, pcov = curve_fit(LogiGaussian, x, y, p0=p0)
        N = int((x[-1] - x[0])/finess + 1)
        coord = np.linspace(x[0], x[-1], num=N)
        fx = LogiGaussian(coord, *popt)
        
        peak = np.argmax(fx)
        thres = np.min(fx) + (np.max(fx) - np.min(fx)) * threshold / 100
        
        if orientation == 'forward':
            edge = coord[np.argmin(np.abs(fx[:peak+1] - thres))]
        elif orientation == 'backward':
            edge = coord[np.argmin(np.abs(fx[peak:] - thres)) + peak]
#        plt.plot(x, y)
#        plt.plot(coord, LogiGaussian(coord, *p0))
#        plt.plot(coord, LogiGaussian(coord, *popt))
#        plt.plot([edge, edge], [np.min(y), np.max(y)])
#        plt.show()        
    except RuntimeError:
        return None
    return edge


def LGStemEdge(x, y, threshold=100, finess=0.05, orientation='forward'):
    if orientation == 'forward':
        ori = 1
        y0 = y[0]
        c = y[-1]-y[0]
        A = np.min(y) - y[0]
    elif orientation == 'backward':
        ori = -1
        y0 = y[-1]
        c = y[0]-y[-1]
        A = np.min(y) - y[-1]
    mid = x[np.argmin(np.abs(y - (y[0] + y[-1])/2))]   
    mu = x[np.argmin(y)]
    sigma = np.abs(mu - mid)/2
    
    p0 = [mid, y0, ori, c, A, mu, sigma]
    try:
        popt, pcov = curve_fit(LogiGaussian, x, y, p0=p0)
        N = int((x[-1] - x[0])/finess + 1)
        coord = np.linspace(x[0], x[-1], num=N)
        fx = LogiGaussian(coord, *popt)
        peak = np.argmin(fx)
        thres = (np.max(fx) - np.min(fx)) * (100-threshold) / 100 + np.min(fx)
        
        if orientation == 'forward':
            edge = coord[np.argmin(np.abs(fx[peak:] - thres)) + peak]
        elif orientation == 'backward':
            edge = coord[np.argmin(np.abs(fx[:peak+1] - thres))]
             
        plt.plot(x, y)
#        plt.plot(coord, LogiGaussian(coord, *p0))
        plt.plot(coord, LogiGaussian(coord, *popt))
        plt.plot([edge, edge], [np.min(y), np.max(y)])
        plt.show()
        
    except:
        return None
    return edge

def StraightLine(coef, x):
    """Straight line y = coef[0] * x + coef[1]
    """
    return coef[0] * x + coef[1]

def WLCRoughness(image, bi_image, low_dist=20, high_dist=350, channel_count=7,
                 downward=300, elevate=5):
    """Edge points along the channels in XSEM
    
    Parameters
    ----------
    bi_image : (N, M) ndarray
        Input binary image.
    low_dist : int
        Low level pixel, start from the etch bottom
    high_dist : int
        High level pixel, start from the etch bottom
    channel_count : int
        Total number of channels
    
    Returns
    -------
    full_channel_points : list
        All points for each side
        
    bots_x : tuple of int
        Coordinate of two x position to draw the bottom limit line in original figure
    
    bots_y : tuple of int
        Coordinate of two y position to draw the bottom limit line in original figure
    
    """
    y_lim, x_lim = bi_image.shape
    # Create edge based on dialation
    edge = BinaryDialateEdge(bi_image)
    # Non-zero counts along the x-axis (row)
    hori_counts = np.sum(edge[downward:,:], axis=1)
    # Approximation of bottom based on the highest horizontal counts along x-axis
    prox_bot = int(downward + np.min(hori_counts.argmax()))
    # Find the closest position (elevate pixel up) that has 2*channel_count horizontal counts
    while True:
        low_channel = prox_bot - elevate
        side_idx = np.nonzero(edge[low_channel,:])[0]
        if len(side_idx) == (channel_count*2):
            break
        elevate += 1
    bot_center_x = np.zeros(channel_count)
    section_enlarge = 1
    channel_section = [None for _ in range(channel_count)]
    channel_width = [None for _ in range(channel_count)]
    max_width = 0
    for i in range(channel_count):
        bot_center_x[i] = 0.5 * (side_idx[i*2] + side_idx[i*2+1])
        channel_width[i] = side_idx[i*2+1] - side_idx[i*2]
        if channel_width[i] > max_width:
            max_width = channel_width[i]
        channel_section[i] = [int(bot_center_x[i] - channel_width[i] * section_enlarge), 
                       min(x_lim-1, int(bot_center_x[i] + channel_width[i] * section_enlarge))]
        center_intens = edge[low_channel:, int(bot_center_x[i])]
        
    center_intens =  edge[low_channel:, bot_center_x.astype(int)]
    bot_relative_y, no_use = np.nonzero(center_intens)
    # Center y of the bottom and middle point of each channel
    bot_center_y = bot_relative_y + low_channel
    # Fit the bottom of each channel with straight line
    bot_slop, bot_intercept = np.polyfit(bot_center_x, bot_center_y, 1)
    # Now start to move the fitted line upward to intercept with each channel
    
    # Intercept of all the lines that are going to use. They share  the same 
    # slop as the bottom line

    full_channel_points = [[] for _ in range(channel_count*2)]   
    """
    The following block is to obtain all the boundary points using line by line smooth
    """   
    for i in range(channel_count):
        lvl_center_y = bot_center_y[i] - low_dist
        top_limit = bot_center_y[i] - high_dist
        section_left_edge, section_right_edge = channel_section[i]
        lvl_width = max_width       
        section_left_edge = max(0, int(bot_center_x[i] - lvl_width * section_enlarge))
        section_right_edge = min(x_lim-1, int(bot_center_x[i] + lvl_width * section_enlarge))
        while (lvl_center_y >= top_limit):
            smooth = AnisotropicFilter1D(image[lvl_center_y, section_left_edge:section_right_edge],
                                                         200, 6000, delta_t=0.3)
#            print 'channel %i at %i level with left at %i and right at %i' %(i, lvl_center_y, section_left_edge,section_right_edge)
            peak = LocalMaxima(np.abs(np.gradient(smooth)), width=int(lvl_width*0.5), highPeak=True)
            if len(peak) != 2:
                print('I got %i maximum using lvl_width of %i'%(len(peak), lvl_width))
                lvl_center_y -= 1
                continue
            x1, x2 = peak
            if (x2 - x1) > lvl_width:
                max_width = x2 - x1
                lvl_width = max_width
#            print 'level width %i' %lvl_width
            x1 += section_left_edge
            x2 += section_left_edge
            section_left_edge = max(0, int(x1 - lvl_width * section_enlarge))
            section_right_edge = min(x_lim-1, int(x2 + lvl_width * section_enlarge))           
            lvl_center_y -= 1
            full_channel_points[i*2].append([x1, lvl_center_y])
            full_channel_points[i*2+1].append([x2, lvl_center_y])      
    bots_x = np.array([0, x_lim])
    bots_y = StraightLine([bot_slop, bot_intercept], bots_x)
    return full_channel_points, bots_x, bots_y
        
    """
    The following block is to obtain all boundary points using the already obtained binary image
    """
    # This is to remember for a side if an edge point at this level exist already
    # side_lvl_record = np.array([[False for _ in xrange(y_lim)] for _ in xrange(channel_count*2)])
#    intercept_range = bot_intercept - np.arange(low_dist, high_dist)
#    channel_center = np.array([bot_center_x, bot_center_y-low_dist])
#    intercept_range = bot_intercept - np.arange(low_dist, high_dist)
#    x_pos = np.arange(x_lim)

#    for intercept in intercept_range:
#        points = [[] for _ in xrange(channel_count*2)]
#        y_pos = StraightLine([bot_slop, intercept], x_pos)
#        for (x, y) in zip(x_pos, y_pos):
#            y = int(y)
#            while y >= y_lim:
#                y -= 1
#            if edge[y, x] == 1:
#                insert_idx = np.searchsorted(channel_center[0], x)
#                if insert_idx == 0: # left most edge
##                    if not side_lvl_record[0][y]:
##                        side_lvl_record[0][y] = True
#                    points[0].append([x, y])
#
#                elif insert_idx == channel_count: # right most edge
##                    if not side_lvl_record[-1][y]:
##                        side_lvl_record[-1][y] = True
#                    points[-1].append([x, y])
#                elif abs(x-channel_center[0][insert_idx]) > abs(x-channel_center[0][insert_idx-1]):
##                    if not side_lvl_record[insert_idx*2-1][y]:
##                        side_lvl_record[insert_idx*2-1][y] = True
#                    points[insert_idx*2-1].append([x, y])
#                else:
##                    if not side_lvl_record[insert_idx*2][y]:
##                        side_lvl_record[insert_idx*2][y] = True
#                    points[insert_idx*2].append([x, y])
#        
#        for i in xrange(channel_count):
#            full_channel_points[2*i] += points[2*i]
#            full_channel_points[2*i+1] += points[2*i+1]
#            channel_center[0][i] = 0.5 * (points[2*i][0][0] + points[2*i+1][0][0])
#    bots_x = np.array([0, x_lim])
#    bots_y = StraightLine([bot_slop, bot_intercept], bots_x)
#    return full_channel_points, bots_x, bots_y

def LocalMaxima(arr, width=100, highPeak=False):
    """Maxima within +/- width
    """
    width = int(width)
    full_max = np.max(arr)
    maxima = (np.diff(np.sign(np.diff(arr))) < 0).nonzero()[0] + 1
    length = len(arr)
    loc_maxima = []
    for m in maxima:
        local_judge = arr[m] > np.max(arr[max(0,m-width):m]) and \
            arr[m] > np.max(arr[m+1:min(m+width, length)])
        if local_judge:           
            if highPeak:
                high_judge =  (abs(arr[m] - full_max)< (0.7 * full_max))
                if high_judge:
                    loc_maxima.append(m)
            else:
                loc_maxima.append(m)
    return np.array(loc_maxima)

def WLCBotCD(image, channel_count, measured_lvl):
    y_lim, x_lim = image.shape   
    thres, _intes, _dens = KernelThresh(image[300:310, :], intens=[0, 40000], 
                                      num=4000, bandwidth=2000,)
#    max_count, _ax, _fig = SEMTool.PixDistribution(image)
    
    bi_image = BinaryConverter(image, thres=thres, scale=1., iteration=3, remove_label=False)
    edge = BinaryDialateEdge(bi_image)
    
    downward = 500
    elevate = 5
    hori_counts = np.sum(edge[downward:,:], axis=1)
    
    # Approximation of bottom based on the highest horizontal counts along x-axis
    prox_bot = int(downward + np.min(hori_counts.argmax()))
    # Find the closest position (elevate pixel up) that has 2*channel_count horizontal counts
    while True:
        low_channel = prox_bot - elevate
        side_idx = np.nonzero(edge[low_channel,:])[0]
        if len(side_idx) == (channel_count*2):
            break
        elevate += 1
    bot_center_x = [None for _ in range(channel_count)]
    bot_center_y = [None for _ in range(channel_count)]
    channel_width = [None for _ in range(channel_count)]
    channel_section = [None for _ in range(channel_count)]
    channel_cornor = [None for _ in range(channel_count*2)]
    channel_center = [None for _ in range(channel_count)]
    
    section_enlarge = 0.9
    for i in range(channel_count):
        bot_center_x[i] = int(0.5 * (side_idx[i*2] + side_idx[i*2+1]))
        channel_width[i] = side_idx[i*2+1] - side_idx[i*2]
        channel_section[i] = [int(bot_center_x[i] - channel_width[i] * section_enlarge), 
                       min(x_lim-1, int(bot_center_x[i] + channel_width[i] * section_enlarge))]
        center_intens = edge[low_channel:, bot_center_x[i]]
        bot_center_y[i] = np.nonzero(center_intens)[0] + low_channel
    
    side_points = [[] for _ in range(channel_count * 2)]
    
    # Find all the edge points around the interface
    recess = 50
    for i in range(channel_count):
        section_left_edge = channel_section[i][0]
        section_right_edge = channel_section[i][1]
        offset = 1
        while True:
            end = bot_center_y[i] - offset
            side_idx = np.nonzero(edge[end, section_left_edge:section_right_edge])[0]
            if len(side_idx) == 2:
                break
            offset += 1
        
        for j in np.arange(end-recess, end):
            x1, x2 = section_left_edge + np.nonzero(edge[j, section_left_edge:section_right_edge])[0]
            if x2 - x1 > channel_width[i]:
                channel_width[i] = x2 - x1
            side_points[i*2].append([x1, j])
            side_points[i*2+1].append([x2, j])
        channel_section[i] = [int(bot_center_x[i] - channel_width[i] * section_enlarge), 
                       min(x_lim-1, int(bot_center_x[i] + channel_width[i] * section_enlarge))]
    
    # Next is to find the interface by locating the "corner" of the image (start of recess)
    for i in range(channel_count):
        left = i*2
        right = i*2+1
        coord = np.array(side_points[left])
        # Note here the y in figure is used as x in the fitting and x in figure is y
        coef, res = OneDCornerDetection(coord[:, 1], coord[:, 0])
        cornor_x, cornor_y, slope = coef
        channel_cornor[left] = [int(cornor_x), int(cornor_y)]
        
        coord = np.array(side_points[right])
        # Note here the y in figure is used as x in the fitting and x in figure is y
        coef, res = OneDCornerDetection(coord[:, 1], coord[:, 0])
        cornor_x, cornor_y, slope = coef
        channel_cornor[right] = [int(cornor_x), int(cornor_y)]
        channel_center[i] = [(channel_cornor[left][0] + channel_cornor[right][0])/2,
                      (channel_cornor[left][1] + channel_cornor[right][1])/2]
    
    channel_cornor = np.array(channel_cornor)
    interface_slop, interface_intercept = np.polyfit(channel_cornor[:,0], channel_cornor[:,1], 1)
    interface_y = [None for _ in range(channel_count)]
    for i in range(channel_count):
        interface_y[i] = StraightLine([interface_slop, interface_intercept], channel_center[i][0])
    
    """
    The following block is to find edge of each line segment, by looking at the
    position of maximum gradient
    """
    CD_points = [[None for _ in range(channel_count*2)] for _ in range(len(measured_lvl))]
    for i in range(len(measured_lvl)):
        for j in range(channel_count):
            lvl_center_y = interface_y[j] - measured_lvl[i]
            lvl_center_y = int(np.round(lvl_center_y))
            section_left_edge = channel_section[j][0]
            section_right_edge = channel_section[j][1]
            smooth = AnisotropicFilter1D(image[lvl_center_y, section_left_edge:section_right_edge],
                                                 200, 5000, delta_t=0.3)
            x1, x2 = LocalMaxima(np.abs(np.gradient(smooth)), width=int(channel_width[j]*0.6))
            CD_points[i][2*j] = [section_left_edge+x1, lvl_center_y]
            CD_points[i][2*j+1] = [section_left_edge+x2, lvl_center_y]            
    return CD_points, interface_slop, interface_intercept

        
    """
    The following block is to find all the edge points based on the previously 
    generated edge image
    """
    #x_pos = np.arange(x_lim)
    #CD_points = [[None for _ in xrange(channel_count*2)] for _ in xrange(len(measured_lvl))]
    #for i in xrange(len(measured_lvl)):
    #    intercept = interface_intercept - measured_lvl[i]
    #    y_pos = SEMTool.StraightLine([interface_slop, intercept], x_pos)
    #    for (x, y) in zip(x_pos, y_pos):
    #        y = int(y)
    #        while y >= y_lim:
    #            y -= 1
    #        if edge[y, x] == 1:
    #            insert_idx = np.searchsorted(bot_center_x, x)
    #            if insert_idx == 0: # left most edge
    #                CD_points[i][0] = [x, y]
    #            elif insert_idx == channel_count: # right most edge
    #                CD_points[i][-1] = [x, y]
    #            elif abs(x-bot_center_x[insert_idx]) > abs(x-bot_center_x[insert_idx-1]):
    #                CD_points[i][insert_idx*2-1] = [x, y]
    #            else:
    #                CD_points[i][insert_idx*2] = [x, y]
    #fig = plt.figure(figsize=(8,8))
    #plt.imshow(part, cmap=plt.cm.gray)
    #for i in xrange(len(measured_lvl)):
    #    for j in xrange(channel_count):
    #        x1, y1 = CD_points[i][j*2]
    #        x2, y2 = CD_points[i][j*2+1]
    #        plt.plot([x1, x2], [y1, y2])
    #        print 'Critical dimension is: %.2f' %(np.sqrt((x1-x2)**2 + (y1-y2)**2) * calib)
    #plt.plot(x_pos, SEMTool.StraightLine([interface_slop, interface_intercept], x_pos), linewidth=0.1)
    #plt.savefig('CD.png', dpi=2000)
    
    """
    This Conclude the process to get edge points based on the total edge image
    """

def TriLineBend(x, x1, y1, x2, y2, x3, y3, x4, y4):
    if x <= x2:
        return LineByTwoPoints(x, x1, y1, x2, y2)
    elif x >= x3:
        return LineByTwoPoints(x, x3, y3, x4, y4)
    else:
        return LineByTwoPoints(x, x2, y2, x3, y3)

vTriLinBend = np.vectorize(TriLineBend)

def TriLineBendFunction(coef, x, x1, x4):
    """
    Start and end are fixed.
    """
    y1, x2, y2, x3, y3, y4 = coef
    return vTriLinBend(x, x1, y1, x2, y2, x3, y3, x4, y4)

def TriLineFit(x, y, x1, x4, beta0):
    # Create a RealData object using input data
    data = spyodr.RealData(x, y)
    # Create linear model for fitting.
    linear_model = spyodr.Model(TriLineBendFunction, extra_args=(x1, x4))
    # Set up ODR with the model and data.
    odr = spyodr.ODR(data, linear_model, beta0=beta0)
    out = odr.run()
    coef = out.beta
    return coef, out.res_var

def LineByTwoPoints(x, x1, y1, x2, y2):
    y = ((x - x1) * y2 - (x - x2) * y1) / (x2 - x1)
    return y


def OneDCornerDetection(x, y):
    """
    More explaination about odr can be found here:
        https://docs.scipy.org/doc/scipy/reference/odr.html#id1
        http://stackoverflow.com/questions/22670057/linear-fitting-in-python-with-uncertainty-in-both-x-and-y-coordinates
        
    """
    # Create a RealData object using input data
    data = spyodr.RealData(x, y)
    # Create linear model for fitting.
    beta0 = [min(y), 0.5*(min(x)+max(x)), 0.707]
    linear_model = spyodr.Model(OneDCornerFunction)
    # Set up ODR with the model and data.
    odr = spyodr.ODR(data, linear_model, beta0=beta0)
    out = odr.run()
    coef = out.beta
    return coef, out.res_var

def OneDCornerFunction(coef, x):
    const, dis_continu, slope = coef
    return const + (x - dis_continu) * slope * (x > dis_continu)


def MonotonFind(arr, target):
    return np.where(np.diff(np.sign(arr - target)))[0]


def WLCBotCD_Manu(image, channel_count, measured_lvl, scan=20, vert_height=0.8, 
                  noise_level=5000, iteration=10, downward=500, elevate=5, 
                  section_enlarge=0.9, recess_search=50):
    """Auto CD measurement for WLC bottom
    
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    channel_count : int
        How many channels exist
    measured_lvl : (N,1) ndarray
        Array of level to be measured
    scan : int, optional
        Just like the "Scan to average" parameter in Quartz PCI
    vert_height : float, between 0 and 1
        Just like the "Vertical Height" parameter in Quartz PCI
    noise_level : int
        Estimated noise level used as parameter for 1D diffusion filter
    downward : int, optional
        From which pixel to search for the bottom of the channels
    elevate : int, optional
        From how many pixel above start to search for edge of channel bottom
    recess_search : int, optional
        Estimate of how far to go to search for the corner. Idealy to be double
        of the true recess height
    
    Returns
    -------
    full_channel_points : list
        All points for each side

    """
    y_lim, x_lim = image.shape   
    thres = KernelThresh(image[300:310, :], intens=[0, 40000], num=4000, bandwidth=2000)
    
    if thres == None:
        thres = 'otsu'   
    bi_image = BinaryConverter(image, thres=thres, scale=1., iteration=3, remove_label=False)
    edge = BinaryDialateEdge(bi_image)
    
    # Use this to find approximated channel bottom. Assume the channel has relatively
    # flat bottom
    hori_counts = np.sum(edge[downward:,:], axis=1)
    
    # Approximation of bottom based on the highest horizontal counts along x-axis
    prox_bot = int(downward + np.min(hori_counts.argmax()))
    
    # Find the closest position (elevate pixel up) that has 2*channel_count horizontal counts
    while True:
        low_channel = prox_bot - elevate
        side_idx = np.nonzero(edge[low_channel,:])[0]
        if len(side_idx) == (channel_count*2):
            break
        elevate += 1
    
    bot_center_x = [None for _ in range(channel_count)]
    bot_center_y = [None for _ in range(channel_count)]
    channel_width = [None for _ in range(channel_count)]
    channel_section = [None for _ in range(channel_count)]
    channel_cornor = [None for _ in range(channel_count*2)]
    

    for i in range(channel_count):
        bot_center_x[i] = int(0.5 * (side_idx[i*2] + side_idx[i*2+1]))
        channel_width[i] = side_idx[i*2+1] - side_idx[i*2]
        channel_section[i] = [int(bot_center_x[i] - channel_width[i] * section_enlarge), 
                       min(x_lim-1, int(bot_center_x[i] + channel_width[i] * section_enlarge))]
        center_intens = edge[low_channel:, bot_center_x[i]]
        bot_center_y[i] = np.nonzero(center_intens)[0][0] + low_channel
    
    side_points = [[] for _ in range(channel_count * 2)]
    
    # Find all the edge points around the interface
    
    # Channel center keeps updating from bottom to top.
    channel_center = [None for _ in range(channel_count)]
    
    for i in range(channel_count):        
        section_left_edge = channel_section[i][0]
        section_right_edge = channel_section[i][1]
        offset = 1
        while True:
            end = bot_center_y[i] - offset
            side_idx = np.nonzero(edge[end, section_left_edge:section_right_edge])[0]
            if len(side_idx) == 2:
                break
            offset += 1
            
        channel_center[i] = bot_center_x[i]
        for j in reversed(np.arange(end-recess_search, end)):
            section_left_edge = channel_section[i][0]
            section_right_edge = channel_section[i][1]
            rough = image[j, section_left_edge:section_right_edge]
            smooth = AnisotropicFilter1D(rough, iteration, noise_level, delta_t=0.3)
            relative_center = channel_center[i] - section_left_edge
#            x1 = np.argmax(smooth[:relative_center]) + section_left_edge
#            x2 = relative_center + np.argmax(smooth[relative_center:]) + section_left_edge
            x1, x2 = section_left_edge + np.nonzero(edge[j, section_left_edge:section_right_edge])[0]
            channel_width[i] = x2 - x1            
            side_points[i*2].append([x1, j])
            side_points[i*2+1].append([x2, j])
            channel_center[i] = int((x2 + x1) * 0.5)           
            channel_section[i] = [int(channel_center[i] - channel_width[i] * section_enlarge), 
                       min(x_lim-1, int(channel_center[i] + channel_width[i] * section_enlarge))]
       
    
    # Next is to find the interface by locating the "corner" of the image (start of recess)
    corner_center = [None for _ in range(channel_count)]
    for i in range(channel_count):
        left = i * 2
        right = i * 2 + 1
        # Deal with left edge first
        coord = np.array(side_points[left])
        # Note here the y in figure is used as x in the fitting and x in figure is y
        coef, res = OneDCornerDetection(coord[:, 1], coord[:, 0])
        cornor_x, cornor_y, slope = coef
        channel_cornor[left] = [int(cornor_x), int(cornor_y)]
        
        # Next deal with right edge
        coord = np.array(side_points[right])
        # Note here the y in figure is used as x in the fitting and x in figure is y
        coef, res = OneDCornerDetection(coord[:, 1], coord[:, 0])
        cornor_x, cornor_y, slope = coef
        channel_cornor[right] = [int(cornor_x), int(cornor_y)]
        
        corner_center[i] = [(channel_cornor[left][0] + channel_cornor[right][0])/2,
                      (channel_cornor[left][1] + channel_cornor[right][1])/2]
    
    channel_cornor = np.array(channel_cornor)
    interface_slop, interface_intercept = np.polyfit(channel_cornor[:,0], channel_cornor[:,1], 1)
    interface_y = [None for _ in range(channel_count)]
    for i in range(channel_count):
        interface_y[i] = StraightLine([interface_slop, interface_intercept], corner_center[i][0])
    
    # height of true recess
    recess_channel = np.array(bot_center_y) - np.array(interface_y)
    """
    The following block is to find edge of each line segment
    """
    CD_points = [[None for _ in range(channel_count*2)] for _ in range(len(measured_lvl))]
    for i in range(len(measured_lvl)):
        for j in range(channel_count):
            lvl_center_y = interface_y[j] - measured_lvl[i]
            lvl_center_y = int(np.round(lvl_center_y))
            
            
            section_left_edge = channel_section[j][0]
            section_right_edge = channel_section[j][1]
            
            scan_avg = np.sum(image[lvl_center_y-scan/2:lvl_center_y+scan/2+1,
                                    section_left_edge:section_right_edge], axis=0) / scan
            # Smooth (by 1D anisotropic diffusion filter) it or not
            smooth = scan_avg
#            smooth = AnisotropicFilter1D(scan_avg, iteration, noise_level, delta_t=0.3)
            
            relative_center = channel_center[j] - section_left_edge
            
            left_peak = np.argmax(smooth[:relative_center])
            right_peak = relative_center + np.argmax(smooth[relative_center:])
            left_edge = (smooth[left_peak] - smooth[relative_center]) *vert_height + smooth[relative_center]
            right_edge = (smooth[right_peak] - smooth[relative_center]) *vert_height + smooth[relative_center]
            
            x1 = left_peak + MonotonFind(smooth[left_peak:relative_center], 
                                         left_edge) + section_left_edge + 1
                                         
            # Need to include the peak point. So right_peak+1
            x2 = relative_center + MonotonFind(smooth[relative_center:right_peak+1], 
                                               right_edge) + section_left_edge    
                                               
            # Update channel width and center for next calculation. Important if 
            # the feature is not straight.
            channel_width[j] = x2 - x1
            channel_center[j] = int((x2 + x1) * 0.5)           
            channel_section[j] = [int(channel_center[j] - channel_width[j] * section_enlarge), 
                       min(x_lim-1, int(channel_center[j] + channel_width[j] * section_enlarge))]
            CD_points[i][2*j] = [x1, lvl_center_y]
            CD_points[i][2*j+1] = [x2, lvl_center_y]            
    return CD_points, recess_channel, interface_slop, interface_intercept