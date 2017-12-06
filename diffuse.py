# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:03:57 2017

@author: LiDo
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tkinter
import tkinter.filedialog
import skimage.external.tifffile as read_tiff
from SSA.analysis import GeneralProcess
from skimage import exposure

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

def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels
    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

   
def ScaleFourth(x, kappa):
    return 1 - np.exp(-3.31488 / (x/kappa**2)**4)

#vScaleFourth = np.vectorize(ScaleFourth)

def ReguIsoNonlinear(image, iteration, kappa, sigma=1, delta_t=0.2):
    """Regularized isotropic nonlinear diffusion
    
    Gaussian regularization
    """
    if len(image.shape) == 1:
        filtered_img = image.astype(float)
        for i in range(iteration):
            smoothed = gaussian_filter(filtered_img, sigma)
            grad = np.gradient(smoothed)
            diff_coef = ScaleFourth(grad**2, kappa)
            delta_img = np.zeros(filtered_img.shape)
            
            coef = diff_coef[:-1] + diff_coef[1:]
            
            image_2_1 = filtered_img[1:] - filtered_img[:-1]
            
            delta_img[1:] += - 0.5 * coef * image_2_1
            delta_img[:-1] += 0.5 * coef * image_2_1
            delta_img = delta_img
            
            filtered_img += delta_img * delta_t
            
    elif len(image.shape) == 2:
        filtered_img = image.astype(float)
        for i in range(iteration):    
            smoothed = gaussian_filter(filtered_img, sigma)
            axisY_grad, axisX_grad = np.gradient(smoothed)
            grad_sq = axisX_grad**2 + axisY_grad**2
            diff_coef = ScaleFourth(grad_sq, kappa)
            
            delta_img = np.zeros(filtered_img.shape)
            
            image_02_01 = filtered_img[:, 1:] - filtered_img[:,:-1]
            image_20_10 = filtered_img[1:, :] - filtered_img[:-1, :]
            
            coef_1 = diff_coef[:, 1:] + diff_coef[:, :-1]
            coef_2 = diff_coef[1:, :] + diff_coef[:-1, :]
            
            delta_img[:, :-1] += 0.5 * coef_1 * image_02_01
            delta_img[:, 1:] += -0.5 * coef_1 * image_02_01
            delta_img[:-1, :] += 0.5 * coef_2 * image_20_10
            delta_img[1:, :] += -0.5 * coef_2 * image_20_10
                            
            filtered_img += delta_img * delta_t
        
    return filtered_img

                        
def EdgeEnhance2D(image, iteration, kappa, sigma=1, delta_t=0.1):
    filtered_img = image.astype(float)
    for i in range(iteration):
        smoothed = gaussian_filter(filtered_img, sigma)
        D_a, D_b, D_c = DiffTensor(smoothed, kappa)
        delta_img = DiffStep2D(filtered_img, D_a, D_b, D_c)
        filtered_img += delta_img * delta_t             
    return filtered_img         
    
def DiffTensor(image, kappa): 
    grad_1 = 0.5 * (image[1:, 1:] + image[1:, :-1] - image[:-1, 1:] - image[:-1, :-1])
    grad_2 = 0.5 * (image[1:, 1:] + image[:-1, 1:] - image[1:, :-1] - image[:-1, :-1])
    grad_sq = grad_1**2 + grad_2**2
    
    grad_sq[grad_sq==0] = 0.00001
    
    grad = np.sqrt(grad_sq)
    eigen_1 = ScaleFourth(grad_sq, kappa)
    eigen_2 = 1.
    
    e_1 = grad_1 / grad
    e_2 = grad_2 / grad
    D_a = eigen_1 * e_1**2 + eigen_2 * e_2**2
    D_b = eigen_2 * e_1**2 + eigen_1 * e_2**2
    D_c = (eigen_1 - eigen_2) * e_1 * e_2
    return D_a, D_b, D_c

def DiffStep2D(image, D_a, D_b, D_c):
    step = np.zeros(image.shape) 
    A = D_a + D_b + 2 * D_c
    B = D_a + D_b - 2 * D_c
    C = D_a - D_b 
    image_22_11 = image[1:, 1:] - image[:-1, :-1]
    image_21_12 = image[1:, :-1] - image[:-1, 1:] 
    step[:-1, :-1] += (image_22_11 * A + image_21_12 * C)
    step[1:, 1:] += (- image_22_11 * A - image_21_12 * C)
    step[:-1, 1:] += (image_22_11 * C + image_21_12 * B)
    step[1:, :-1] += (- image_22_11 * C - image_21_12 * B)
    step = step * 0.25
    return step

if __name__ == "__main__":
    root = tkinter.Tk()
    path = tkinter.filedialog.askopenfilenames(parent=root,title='Choose a files')
    root.destroy()
    
    image = read_tiff.imread(path)
    
    f=plt.figure(figsize=(10,8))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    
    part = image[:800, :]
#    part = image[:700,:500,]
    #part = image[200:600,:1000,0]
    y_lim, x_lim = part.shape
    f=plt.figure(figsize=(10,8))
    plt.imshow(part, cmap=plt.cm.gray)
    plt.show()
    
    
    
    #from scipy import misc
    #image = misc.imread('noise.png')
    #part = image[:,:,1]
    #f=plt.figure(figsize=(10,10))
    #plt.imshow(part, cmap=plt.cm.gray)
    #plt.show()
    
    #denoise = ReguIsoNonlinear2D(part, 300, 2000, sigma=1)
    #denoise = ReguIsoNonlinear2D(part, 300, 3, sigma=3, delta_t=0.1)
    
    
    
    
    denoise = EdgeEnhance2D(part, 600, 3.2, sigma=1., delta_t=0.1)
    #denoise = EdgeEnhance2D(part, 300, 1000, sigma=2.5, delta_t=0.1)
    
    f=plt.figure(figsize=(10,8))
    plt.imshow(denoise, cmap=plt.cm.gray)
    plt.show()
    
    bi_img = GeneralProcess.BinaryConverter(denoise, thres=175)
    edge = GeneralProcess.BinaryDialateEdge(bi_img)
    y, x = np.nonzero(edge)
    f = plt.figure(figsize=(10, 8))
    plt.imshow(part, cmap=plt.cm.gray)
    plt.plot(x, y, 'r.', markersize=1)
    plt.show()
    
    
    a1, b1 = exposure.histogram(part)
    a2, b2 = exposure.histogram(denoise)
    plt.plot(b1, a1, label='Original')
    plt.plot(b2, a2, label='After Filtering')
    plt.legend(loc=2)
    plt.show()