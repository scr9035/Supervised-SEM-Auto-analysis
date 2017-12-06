# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li


import numpy as np
import skimage.external.tifffile as read_tiff
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from skimage.segmentation import active_contour
import skimage.filters as skifilters
from skimage import img_as_ubyte
import skimage.exposure as skiexpos
import skimage.transform as skitransform


from scipy import ndimage, stats
import skimage.morphology as skimorph
import skimage.measure as skimeasure
from skimage import exposure

import tkinter
import tkinter.filedialog

import scipy.stats as spystat
from sklearn.neighbors import KernelDensity
import sklearn.mixture as skMixture
from sklearn.cluster import spectral_clustering

import scipy.odr as spyodr
import scipy.signal as spysig
import matplotlib as mpl
import sklearn.cluster as skcluster
import os

from scipy.optimize import curve_fit
from SSA.analysis.channel import VertiChannel, IntensInterface, ChannelEdgeIntens
from SSA.analysis.GeneralProcess import (PixDistribution, GaussianMixThres, 
                        BinaryConverter, BinaryDialateEdge, BinaryErosionEdge,
                        AnisotropicImageFilter1D, AnisotropicFilter2D, 
                        AnisotropicFilter1D, LGSemEdge, KernelThresh,
                        vTriLinBend, BiGaussCDFThres)

def BimodalThres(image, bin_num=1000):
    values, bins = np.histogram(image, bins=bin_num)
    idx = np.nonzero(values)
    count = values[idx]
    intensity = bins[:-1][idx]
    width = int(len(count)/6)
    peak_idx = spysig.find_peaks_cwt(count, np.arange(3,width))
    combine = [(count[i], i, intensity[i]) for i in peak_idx]
    combine.sort(reverse=True)
    thres = (combine[0][2] + combine[1][2])/2    
    return thres
# In[]
# Read images

root = tkinter.Tk()
path = tkinter.filedialog.askopenfilenames(parent=root,title='Choose a files')
root.destroy()
pic_name = os.path.basename(path[0])

image = read_tiff.imread(path)
#part = image[120:,:]
part = image[200:,:,0]
y_lim, x_lim = part.shape
#
plt.imshow(part, cmap=plt.cm.gray)
plt.show()

VertiChannel(part, 100, quality='uneven')





# In[] random walker segmentation

#from skimage.segmentation import random_walker
#from skimage.data import binary_blobs
#from skimage.exposure import rescale_intensity
#import skimage
#
#markers = np.zeros(part.shape, dtype=np.uint)
#markers[part < 9000] = 1
#markers[part > 61000] = 2
#
## Run random walker algorithm
#labels = random_walker(part, markers, beta=10, mode='bf')
#
## Plot results
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10),
#                                    sharex=True, sharey=True)
#ax1.imshow(part, cmap='gray', interpolation='nearest')
#ax1.axis('off')
#ax1.set_adjustable('box-forced')
#ax1.set_title('Noisy data')
#ax2.imshow(markers, cmap='magma', interpolation='nearest')
#ax2.axis('off')
#ax2.set_adjustable('box-forced')
#ax2.set_title('Markers')
#ax3.imshow(labels, cmap='gray', interpolation='nearest')
#ax3.axis('off')
#ax3.set_adjustable('box-forced')
#ax3.set_title('Segmentation')
#
#fig.tight_layout()
#plt.show()

# In[]


#f = plt.figure(figsize=(10, 10))
#plt.imshow(part)
#plt.colorbar()
#plt.show()

#high = cdf_centers[np.argmin(np.abs(img_cdf - 0.8))]
#low = cdf_centers[np.argmin(np.abs(img_cdf - 0.2))]
#
#cutoff = cdf_centers[np.argmin(np.abs(img_cdf - 0.9))]
#data = part.flatten()
#data = np.ma.masked_where(data>=high, data)
#img_cdf, cdf_centers = exposure.cumulative_distribution(data[~data.mask])
#plt.plot(cdf_centers, img_cdf)
#plt.show()
#img_cdf, cdf_centers = exposure.cumulative_distribution(part)
#plt.plot(cdf_centers, img_cdf)
#plt.show()


#part[part >= high] = 2
#part[(part >= low) & (part < high)] = 1
#part[(part > 10) & (part < low)] = 0
#
#f = plt.figure(figsize=(10, 10))
#plt.imshow(part)
#plt.colorbar()
#plt.show()

#
#ref = IntensInterface(part)
#below = 200

#hist, bin_centers = exposure.histogram(part[below:,:])
#plt.plot(bin_centers, hist)
#plt.show()
#compress = np.sum(part[below:, :], axis=0)/(y_lim-below)
#img_cdf, cdf_centers = exposure.cumulative_distribution(compress)
#
#peak_value = bin_centers[np.argmax(hist)]
#peak_percent = img_cdf[np.argmin(np.abs(cdf_centers- peak_value))]
#x1 = np.min(cdf_centers)
#y1 = 0
#x2 = (peak_value + x1) / 2
#y2 = peak_percent / 2
#x4 = np.max(cdf_centers)
#y4 = 1
#x3 = (peak_value + x4) / 2
#y3 = (peak_percent + 1) / 2
#
#beta0 = [x1, y1, x2, y2, x3, y3, x4, y4]
#
#def TriLineBendFunction(coef, x):
#    x1, y1, x2, y2, x3, y3, x4, y4 = coef
#    return vTriLinBend(x, x1, y1, x2, y2, x3, y3, x4, y4)
#
#def TriLineFit(x, y, beta0):
#    """
#    More explaination about odr can be found here:
#        https://docs.scipy.org/doc/scipy/reference/odr.html#id1
#        http://stackoverflow.com/questions/22670057/linear-fitting-in-python-with-uncertainty-in-both-x-and-y-coordinates
#        
#    """
#    # Create a RealData object using input data
#    data = spyodr.RealData(x, y)
#    # Create linear model for fitting.
#    linear_model = spyodr.Model(TriLineBendFunction)
#    # Set up ODR with the model and data.
#    odr = spyodr.ODR(data, linear_model, beta0=beta0)
#    out = odr.run()
#    coef = out.beta
#    return coef, out.res_var
#
#coef, _ = TriLineFit(cdf_centers, img_cdf, beta0)
#
#plt.plot(cdf_centers, img_cdf)
#plt.plot(cdf_centers, TriLineBendFunction(coef, cdf_centers))
#plt.show()


#hist, bin_centers = exposure.histogram(compress)


#thres2 = GaussianMixThres(part[below:,:], means_init=[[np.min(bin_centers)], [bin_centers[np.argmax(hist)]]], n_init=3, scale=0.5)
#thres = GaussianMixThres(part[below:, :], scale=0)
#thres = coef[2]
#plt.plot(bin_centers, hist)
#plt.vlines(thres, 0, 25)
#plt.show()
#
#plt.plot(compress)
#plt.hlines(thres,0,1024)
#plt.show()



# In[]
# Read .tif tags!
#
#import tifffile
#with tifffile.TiffFile(path[0]) as tif:
#    images = tif.asarray()
#    for page in tif:
#        for tag in page.tags.values():
#            if tag.name == 'helios_metadata':
#                if 'Scan' in tag.value:
#                    print(tag.value['Scan']['PixelHeight'])
#            if tag.name == 'sem_metadata':
#                if '' in tag.value:
#                    print(tag.value[''][3])
#            t = tag.name, tag.value
#            print(t)
#            image = page.asarray()


# In[]
## Experimenting proper threshold

#channel_count, channel_center, plateau = VertiChannel(part, 0, target='dark')
#
#ref1 = IntensInterface(part, ref_range=[10,np.Inf])
#ref2 = IntensInterface(part, ref_range=[10,int(y_lim/2)])
#
#fig = plt.figure(figsize=(8,8))
#plt.imshow(part, cmap=plt.cm.gray)
#for plat in plateau:
#    plt.plot([plat, plat], [0, y_lim], 'r')
#for center in channel_center:
#    plt.plot([center, center], [0, y_lim], 'y')
#
#plt.plot([0, x_lim], [ref1, ref1], 'b')
#plt.plot([0, x_lim], [ref2, ref2], 'r')
#plt.show()


#lvl = 200
#threshold = 100
#for i in range(2):
#    center = int(channel_center[i])
#    left = int(plateau[i])
#    right = int(plateau[i+1])
#    
#    x1 = np.arange(left, center)
#    y1 = part[lvl, left:center]
#    y1 = np.array(y1, dtype='int64')
#    left_edge = LGSemEdge(x1, y1, threshold=threshold, orientation='backward')
#    
#    x2 = np.arange(center, right)
#    y2 = part[lvl, center:right]
#    y2 = np.array(y2, dtype='int64')
#    right_edge = LGSemEdge(x2, y2, threshold=threshold, orientation='forward')



#plt.plot(part[600,250:350])
#plt.show()
#
#part = AnisotropicImageFilter1D(part, 10, 5, delta_t=0.3)
#plt.plot(part[600,250:350])
#plt.show()
#
#max_count, ax, fig = PixDistribution(part)
#gmix = skMixture.GaussianMixture(n_components=3, n_init=5, 
#                                     init_params='kmeans')
#gmix.fit(part.flatten()[:, np.newaxis])
#mean1, mean2, mean3 = gmix.means_[:,0]
#mean1, mean2, mean3 = sorted([mean1, mean2, mean3])
#plt.plot([mean1, mean1], [0, max_count], 'r')
#plt.plot([mean2, mean2], [0, max_count], 'b')
##plt.plot([mean3, mean3], [0, max_count])
##thres = GaussianMixThres(part)
#thres = (mean1 + mean2)/2
#plt.plot([thres, thres], [0, max_count])
#plt.show()


#bi_fig = BinaryConverter(part, thres=thres)

#plt.imshow(bi_fig, cmap=plt.cm.gray)
#plt.show()

#boundary = BinaryDialateEdge(bi_fig)
#boundary = BinaryErosionEdge(bi_fig)

#fig = plt.figure(figsize=(12,12))
#plt.imshow(boundary, cmap=plt.cm.gray)
#plt.show()
#plt.savefig('boundary.png', dpi=2000)


#coord = np.nonzero(boundary)
#props = dict(marker='+', markersize=0.1, color='r', mfc='w', ls='none',
#                     alpha=1, visible=True)
#y, x = coord
#lines = mpl.lines.Line2D(x, y, **props)
#fig, ax = plt.subplots()
#ax.add_line(lines)
#plt.imshow(part, cmap=plt.cm.gray)
#plt.show()
#plt.savefig('boundary.png', dpi=2000)