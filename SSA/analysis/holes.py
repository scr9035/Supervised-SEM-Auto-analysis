# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

import skimage.morphology as skimorph
import skimage.measure as skimeasure
import skimage.transform as skitransform
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

from .GeneralProcess import SegmentIntersect, Intersection

def TopGeometryAnalysis(image, shape='ellip'):
    """ Return the fitted geometry information of the given image
    
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    shape : str, optional
        Geometry shape used to fit regions of image. If 'ellip' (default), fit 
        the regions with an ellipse. If 'rect', fit the regions with a rectangle
    edge_remove : boolean
        Remove the regions that are too close to the edge of image
    area_thres : float
        Percentage of the whole image as threshold to remove obviously wrong region
    
    Returns
    -------
    properties : python dictionary
        Store properties of all labeled regions
    fig : matplotlib figure class
        Figure where the axis is
    ax : matplotlib axis class
        axis of the figure
    
    """
    labeled_image = skimorph.label(image)
    regions = skimeasure.regionprops(labeled_image)
#    properties = {'area' : [],
#                  'major' : [],
#                  'minor' : [],
#                  'orientation' : [],
#                  'center' : [],
#                  'eccentricity' : [],
#                  'count' : 0}
    return regions

#    max_row, max_col = image.shape
#    for i in range(len(regions)):
#        prop = regions[i]
#        minr, minc, maxr, maxc = prop.bbox
#        if minc <= 1 or minr <= 1 or maxc >= max_col-2 or maxr >= max_row-2:
#            continue
#        if prop.area < max_row * max_col * area_thres / 100.0:
#            continue
#        y0, x0 = prop.centroid
#        properties['area'].append(prop.area)
#        properties['major'].append(prop.major_axis_length)
#        properties['minor'].append(prop.minor_axis_length)
#        properties['orientation'].append(prop.orientation) # Unit is radius, clock wise
#        properties['center'].append([x0, y0])
#        properties['eccentricity'].append(prop.eccentricity)
#        properties['count'] += 1
        
#        x1 = x0 + math.cos(orientation) * 0.5 * prop.major_axis_length
#        y1 = y0 - math.sin(orientation) * 0.5 * prop.major_axis_length
#        x2 = x0 - math.sin(orientation) * 0.5 * prop.minor_axis_length
#        y2 = y0 - math.cos(orientation) * 0.5 * prop.minor_axis_length
#        major = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
#        minor = np.sqrt((x2 - x0)**2 + (y2 - y0)**2)
#        
#        if shape == 'ellip':        
#        # Here angle has unit: degree, anti-clock wise
#            fitted_obj = matplotlib.patches.Ellipse([x0, y0], width=2*major, 
#                                                   height=2*minor, 
#                                                   angle=(2*np.pi - orientation) * 180/np.pi,
#                                                   fill=False,
#                                                   edgecolor='red',
#                                                   linestyle='-',
#                                                   linewidth=1)
#        elif shape == 'rect':
#            fitted_obj = matplotlib.patches.Rectangle((minc, minr),
#                                      maxc - minc,
#                                      maxr - minr,
#                                      fill=False,
#                                      edgecolor='red',
#                                      linewidth=1)
        
        
    return properties

def GridMatching(image_fit, grid='rect', remove_label=True, open_limit=40, angle_diff=2):
    """Find grid of features and count capping
    
    Parameters
    ----------
    image : (N, M) ndarray
        Input binary image
       
    Returns
    -------
    miss_point : list of tuples
        The coordinates of points that are missed
    """
    
    y_lim, x_lim = image_fit.shape
    
    labeled_image = skimorph.label(image_fit)
    regions = skimeasure.regionprops(labeled_image)
    open_count = len(regions)
    if open_count < open_limit:
        return open_count, 0, 0, None, None
    
    if grid == 'rect':
        verti_search = np.linspace(-angle_diff, angle_diff, num=101) * np.pi / 180
        v_space, v_theta, v_d = skitransform.hough_line(image_fit, theta=verti_search)        
        vert_accu, vert_angles, vert_dists = skitransform.hough_line_peaks(v_space, v_theta, v_d, 
                                min_distance=30, threshold=np.max(v_space)/15.)
        
        vert_para = list(zip(vert_dists, vert_angles, vert_accu))
        vert_para.sort()
        vert_lines_set = []            
        last_accu = None
        last_dist = None
        for dist, angle, accu in vert_para:
            if dist < x_lim * 0.02 or abs(dist - x_lim) < x_lim * 0.02:
                continue
            x0 = dist / np.cos(angle)
            x1 = - y_lim * np.tan(angle) + x0
            line = np.array([[x0, 0], [x1, y_lim]])           
            if len(vert_lines_set) > 0:
                if SegmentIntersect(line, vert_lines_set[-1]):
                    if accu > last_accu:
                        vert_lines_set[-1] = line
                        last_accu = accu
                        last_dist = dist                       
                elif abs(dist-last_dist) < 0.01 * x_lim:
                    if accu > last_accu:
                        vert_lines_set[-1] = line
                        last_accu = accu
                        last_dist = dist
                else:
                    vert_lines_set.append(line)
                    last_accu = accu
                    last_dist = dist
            else:
                vert_lines_set.append(line)
                last_accu = accu
                last_dist = dist
                
        # Calculate horizontal lines
        hori_search = np.linspace(90-angle_diff, 90+angle_diff, num=101) * np.pi / 180
        h_space, h_theta, h_d = skitransform.hough_line(image_fit, theta=hori_search) 
        hori_accu, hori_angles, hori_dists = skitransform.hough_line_peaks(h_space, h_theta, h_d, 
                                min_distance=30, threshold=np.max(h_space)/15.)
        hori_para = list(zip(hori_dists, hori_angles, hori_accu))
        hori_para.sort()
        hori_lines_set = []            
        last_accu = None
        last_dist = None
        for dist, angle, accu in hori_para:
            if dist < y_lim * 0.02 or abs(dist - y_lim) < y_lim * 0.02:
                continue
            y0 = dist / np.sin(angle)
            y1 = (dist - x_lim * np.cos(angle)) / np.sin(angle)
            line = np.array([[0, y0], [x_lim, y1]])           
            if len(hori_lines_set) > 0:
                if SegmentIntersect(line, hori_lines_set[-1]):
                    if accu > last_accu:
                        hori_lines_set[-1] = line
                        last_accu = accu
                elif abs(dist-last_dist) < 0.01 * y_lim:
                    if accu > last_accu:
                        hori_lines_set[-1] = line
                        last_accu = accu
                        last_dist = dist
                else:
                    hori_lines_set.append(line)
                    last_accu = accu
                    last_dist = dist
            else:
                hori_lines_set.append(line)
                last_accu = accu   
                last_dist = dist
        line_collection = [hori_lines_set, vert_lines_set]
    else:
        return
    
    cap_point = []
    for line1 in line_collection[0]:
        for line2 in line_collection[1]:
            cross_point = Intersection(line1, line2)
            imag_r = int(np.floor(cross_point[1]))
            imag_c = int(np.floor(cross_point[0]))
            if imag_r >=y_lim or imag_c >= x_lim or imag_r < 0 or imag_c < 0:
                continue
            elif image_fit[imag_r, imag_c] == 0:
                cap_point.append(cross_point)    
    cap_point = np.array(cap_point)
    cap_count = len(cap_point)
    tot_count = len(line_collection[0]) * len(line_collection[1])   
    return open_count, cap_count, tot_count, cap_point, line_collection