#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:27:44 2019

@author: rfetick
"""

import numpy as np
from scipy.special import jv


__all__ = ["circarr", "binning", "airy", "imcenter", "circavg", "circavgplt", "random_sample"]


#%% CONSTANTS
# Radian to arcsec
RAD2ARCSEC = 180./np.pi * 3600.

#%% FUNCTIONS
def random_sample(psd, L):
    """
    Generate a random screen from the PSD
    
    Parameters
    ----------
    psd : np.array
        The PSD array, with null frequency at the center of the array.
    L : float
        Physical extent of the array.
    """
    psd_sq = np.fft.fftshift(np.sqrt(psd))
    shp = np.shape(psd)
    tab = psd_sq * (np.random.randn(*shp)+1j*np.random.randn(*shp)) / L
    tab = np.real(np.fft.ifft2(tab)) * np.size(psd)
    return tab


def circarr(shape, center=(None, None)):
    """Compute array of radii to the center of array
    
    Parameters
    ----------
    shape : tuple, list (of 2 elements)
        Number of pixels on X and Y coordinates
    
    Keywords
    --------
    center : tuple or list (of 2 elements)
        Coordinates of center
        Default: center of array at [(Nx-1)/2,(Ny-1)/2]
    
    Example
    -------
    >>> Npix = (200,200)
    >>> center = (100,100)
    >>> r = circarr(Npix,center=center)
    
    """
    center = list(center)
    if min(shape) <= 0:
        raise ValueError("You should ensure Nx > 0 and Ny > 0")
    if len(center) != 2:
        raise ValueError("Keyword `center` should be a tuple of 2 elements")
    if center[0] is None:
        center[0] = (shape[0]-1) / 2
    if center[1] is None:
        center[1] = (shape[1]-1) / 2
    xx, yy = np.ogrid[0:shape[0], 0:shape[1]]
    
    return np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)


def binning(image, k):
    """Bin an image by a factor `k`
    
    Example
    -------
    >>> x,y = np.mgrid[0:10,0:10]
    >>> data = (-1)**x * (-1)**y
    >>> data_bin = binning(data,2)
    
    """
    if k==1:
        return np.copy(image)
    
    shp = np.shape(image)
    nx = int(shp[0] / k)
    ny = int(shp[1] / k)
    out = np.zeros((nx, ny))
    for i in range(k):
        for j in range(k):
            out += image[i:k*nx:k, j:k*ny:k]
    return out


def airy(shape, samp, occ=0., center=(None,None)):
    """Create a pupil diffraction pattern for a given sampling
    
    Parameters
    ----------
    shape : tuple, list
        Number of pixels on X and Y
    samp : float
        Sampling = (LAMBDA*FOCAL)/(PIX*D)
    
    Keywords
    --------
    occ : float
        Eventual occultation ratio = D_secondary / D_primary
    center : tuple
        Center on X and Y of the diffraction pattern
    
    Example
    -------
    >>> a = airy((256,256),4,occ=0.2)
    
    """
    if samp <= 0:
        raise ValueError("You should ensure samp > 0")
    if occ < 0 or occ >= 1:
        raise ValueError("You should ensure 0 <= occ < 1")

    rr = circarr(shape,center=center) * np.pi / samp
    index = np.where(rr == 0)
    rr[index] = 1.0
    
    if occ > 0:
        aa = 4.0/((1-occ**2)**2) * ((jv(1,rr) - occ*jv(1,occ*rr))/rr)**2
    else:
        aa = 4.0 * (jv(1,rr)/rr) ** 2

    aa[index] = 1.0
    return aa / aa.sum()


def imcenter(img, size, maxi=True, GC=False, center=None):
    """Center a tabular on its maximum, or on its center of gravity
    
    Parameters
    ----------
    img : numpy.ndarray (dim=2)
        The image to center
    size : tuple or list of 2 ints
        Size of centered image
    
    Keywords
    --------
    maxi : bool
        Centers image on maximum (default = True)
    GC : bool
        Centers image on gravity center (default = False)
    center : tuple or list of 2 ints
        User defined centering on a chosen pixel (default = None)
        
    Returns
    -------
    newim : numpy.ndarray (dim=2)
    
    """

    ### SET to False the default choice, since user wants another option
    if (GC is True) or (center is not None):
        maxi = False

    sX,sY = np.shape(img)

    if len(size) != 2:
        raise ValueError("size keyword should contain 2 elements")
    if (size[0] < 0) or (size[1] < 0):
        raise ValueError("size keyword should contain only positive numbers")

    ### CENTER ON USER DEFINED CENTER
    if center is not None:
        cX,cY = center

    ### CENTER ON MAX, default method
    if maxi:
        index = np.where(img == img.max())
        if len(index[0]) > 1:
            print("Imcenter warning: more than one maximum found")
        cX = index[0][0]
        cY = index[1][0]

    ### CENTER ON GRAVITY CENTER
    if GC:
        x = np.arange(sX)
        y = np.arange(sY)
        cX = int(np.round(((x * img.sum(axis=1)).sum()) / (img.sum())))
        cY = int(np.round(((y * img.sum(axis=0)).sum()) / (img.sum())))

    ### COMMON BLOCK FOR CENTERING!!!
    if (cX - size[0] / 2 < 0) or (cY - size[1] / 2 < 0) or (cX + size[0] / 2 >= sX) or (cY + size[1] / 2 >= sY):
        raise ValueError("Defined size is too big, cannot center image")
    vX = np.arange(cX - size[0] / 2, cX + size[0] / 2, dtype=int)
    vY = np.arange(cY - size[1] / 2, cY + size[1] / 2, dtype=int)
    newim = img[vX][:, vY]

    return newim


def circavg(tab, center=(None, None)):
    """Compute the circular average of a given array
    
    Parameters
    ----------
    tab : numpy.ndarray (dim=2)
        Two-dimensional array to compute its circular average
    
    Keywords
    --------
    center : tuple or list of two elements
        Position of center
    
    Returns
    -------
    vec : numpy.ndarray (dim=1)
        Vector containing the circular average from center
        
    """
    if tab.ndim != 2:
        raise ValueError("Input `tab` should be a 2D array")
    rr = circarr(tab.shape, center=center)
    avg = np.zeros(int(rr.max()), dtype=tab.dtype)
    for i in range(int(rr.max())):
        index = np.where((rr >= i) * (rr < (i + 1)))
        avg[i] = tab[index[0], index[1]].sum() / index[0].size
    return avg


def circavgplt(tab, center=(None, None), dtype=float):
    """Create a (x,y) output ready for plotting where
        `x` are indices [pixel]
        `y` contains the circular average of the 2D `tab`
    Output is symmetrized with respect to x=0 for a symmetric plot"""
    y = circavg(tab, center=center)
    x = np.arange(len(y), dtype=dtype)
    xsym = -x[:0:-1]
    ysym = y[:0:-1]
    return (np.concatenate((xsym, x)), np.concatenate((ysym, y)))
