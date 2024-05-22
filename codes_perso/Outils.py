# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:10:25 2024

@author: cdartevelle
"""

import numpy as np
import matplotlib.pyplot as plt

print("🛠️🪛🏋️ Outils pour LIFT ajoutés : 🏋️🪛🛠️")
print(" - classe Buffer")
print(" - echantPSF(PSF)")
print(" - ft2 et ift2")
print(" - centerPSF(PSF)")
print("")

#%% Classe Buffer

class buffer:
    
    def __init__(self, size:int, placeholder=0):
        """
        Parameters :
        ------------
        size : int, number of elements of the buffer. Cannot be changed once created
        
        Using the buffer :
        ------------------
            buf = buffer(size) to inintialize buffer
            
            buf.add(element) to add an element (will erase the oldest one)
            
            value = buf.access(index) to access an element
             
                index = -1 for the last added element, -2 for the previous
                ... until -size for the oldest element
                
            list = buf.accessAll(mode) to acess all elements
                mode = 'newest2oldest', 'oldest2newest' or 'raw' (internal storage mode)
            
            buf.size to acess the size of the buffer
            
        """
        self.size = size
        self.__data__ = [placeholder] * self.size
        self.__counter__ = -1
        
        
        
    def add(self, element):
        if self.__counter__+1 < self.size:
            self.__counter__ += 1
        elif self.__counter__+1 == self.size:
            self.__counter__ = 0
        else :
            raise ValueError(f"Object is being added outisde of the buffer (idex {self.__counter__} out of {self.size})")
        
        self.__data__[self.__counter__] = element
            
        
        
    def access(self, index:int):
        if index > 0:
            raise ValueError(f"Index must be negative ({index})")
        elif index < -self.size:
            raise ValueError(f"Index must be smaller than the buffer size ({index})")
        else :
            
            if self.__counter__ + index >= 0:
                return self.__data__[self.__counter__ + index]
            elif self.__counter__ + index < 0:
                return self.__data__[self.size + self.__counter__ + index]
            
            
            
    
    def accessAll(self, mode:str='newest2oldest'):
        if mode == 'newest2oldest':
            out = [0] * self.size
            for i in range(self.size):
                out[i] = self.access(-i)
            return out
        
        elif mode == 'oldest2newest':
            out = [0] * self.size
            for i in range(self.size):
                out[self.size - i] = self.access(-i)
            return out 
        
        elif mode == 'raw':
            return self.__data__
        
        else :
            raise ValueError("mode must be set to 'newest2oldest', 'oldest2newest' or 'raw")
            
#%% Echantillonage PSF


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



def circavg(tab, center=(None, None)):
    """Compute the circular average of a given array
    
    Parameters
    ----------
    tab : np.ndarray (dim=2)
        Two-dimensional array to compute its circular average
    
    Keywords
    --------
    center : tuple or list of two elements
        Position of center
    
    Returns
    -------
    vec : np.ndarray (dim=1)
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



def echant(PSF):
    TF = np.fft.fftshift(np.fft.fft2(PSF))
    MTF = np.abs(TF)**2
    plt.plot(circavg(MTF))
    plt.yscale('log')
    plt.xscale('log')



def find_zero(func, epsilon=1e-3):
    """
    Find the moment where a functions hits the cutting frequency
    """
    fc = None
    for i, value in enumerate(func):
        if value <= epsilon:
            fc = i
            break
    for j, _ in enumerate(func[fc:]):
        if func[j+1] > func[j]:
            return fc + j



def find_sampling(psf):
    fto = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))
    curve = circavg(np.abs(fto)**2)
    f0 = find_zero(curve)
    print(len(curve))
    print(f0)

    return len(curve)*2/f0


#%% outils pratiques

def ft2(mat):
    return np.fft.fftshift(np.fft.fft2(mat))

def ift2(mat):
    return np.fft.ifftshift(np.fft.ifft2(mat))


#%% fonctions centre PSF

def cog(img, threshold=0, min_threshold=0, **kwargs):
    """
    Original function from AOtools.imageprocessing
    ---> https://github.com/AOtools/aotools/blob/main/aotools/image_processing/centroiders.py
    
    Centroids an image, or an array of images.
    Centroids over the last 2 dimensions.
    Sets all values under "threshold*max_value" to zero before centroiding
    Origin at 0,0 index of img.

    Parameters:
        img (ndarray): ([n, ]y, x) 2d or greater rank array of imgs to centroid
        threshold (float): Percentage of max value under which pixels set to 0

    Returns:
        ndarray: Array of centroid values (2[, n])

    """

    if threshold != 0:
        if len(img.shape) == 2:
            thres = np.max((threshold*img.max(), min_threshold))
            img = np.where(img > thres, img - thres, 0)
        else:
            thres = np.maximum(threshold*img.max(-1).max(-1), [min_threshold]*img.shape[0])
            img_temp = (img.T - thres).T
            zero_coords = np.where(img_temp < 0)
            img[zero_coords] = 0

    if len(img.shape) == 2:
        y_cent, x_cent = np.indices(img.shape)
        y_centroid = (y_cent*img).sum()/img.sum()
        x_centroid = (x_cent*img).sum()/img.sum()

    else:
        y_cent, x_cent = np.indices((img.shape[-2], img.shape[-1]))
        y_centroid = (y_cent*img).sum(-1).sum(-1)/img.sum(-1).sum(-1)
        x_centroid = (x_cent*img).sum(-1).sum(-1)/img.sum(-1).sum(-1)

    return np.array([x_centroid, y_centroid])



def cogPSF(PSF, search_radius:float=20):
    """
    Description
    -----------
        Finds the center of gravity of the PSf around the brightest point of the
        image, avoiding noise

    Parameters
    ----------
        PSF : np.array
        search_radius : float, search_radius in % of the image size around the
            brightest point. The default is 5.

    Returns
    -------
        coordinates of the center : np.array([float, float])
    """
    
    (maxIx, maxIy) = np.unravel_index(np.argmax(PSF), PSF.shape)
    #radius = PSF.shape[0] * search_radius // 10
    radius = 10
    
    center = np.zeros(PSF.shape)
    center[maxIx-radius:maxIx+radius, maxIy-radius:maxIy+radius] = PSF[maxIx-radius:maxIx+radius, maxIy-radius:maxIy+radius]

    return cog(center)


def centerPSF(PSF, size:(int, int), recur:int=0):
    """
    Desrciption
    -----------
        Finds the center of gravity of the PSF on an image, and centers it
        on a new image of specified size. The new image can be larger or
        smaller than the original one.
    
    Parameters
    ----------
        PSF : np.array to be cropped
        size : (int, int), size of the output image
        recur : [DO NOT USE] inetrenal var to track recursion limit of the function

    Returns
    -------
        centerPSF : np.array of asked size, with the PSF centered on it
    """
    
    recur += 1 #recursion counter
    if recur == 3:
        raise ValueError("Impossible to crop PSF, infinite recursion loop")
    
    PSF[:5, :5] = 0 #removing the data pixels from the camera on the top left corner
    [x, y] = cogPSF(PSF)

    
    x1 = round(x-size[0]/2)
    x2 = round(x+size[0]/2)
    y1 = round(y-size[1]/2)
    y2 = round(y+size[1]/2)
    
    if x1>=0 and y1>=0 and x2<=PSF.shape[0] and y2<=PSF.shape[1]:
        #simple case, juste cropping the original image
        PSF_centered = PSF[y1:y2, x1:x2]
        return PSF_centered
        
    else : 
        #other cases, the new PSF is bigger than than the original
        dx1 = 0
        dx2 = 0
        dy1 = 0
        dy2 = 0
        
        if x1<0 :
            dx1 = -x1
        if y1<0 :
            dy1 = -y1
        if x2>PSF.shape[0] :
            dx2 = x2 - PSF.shape[0]        
        if y2>PSF.shape[1]:
            dy2 = y2 - PSF.shape[1]
        
        larger_PSF = np.pad(PSF, np.max([dx1, dx2, dy1, dy2]))
        return centerPSF(larger_PSF, size, recur)



#%% Ecart entre les PSF

def PSFdisimilarity(PSF1, PSF2, display=False):
    """
    

    Parameters
    ----------
    PSF1 : TYPE
        DESCRIPTION.
    PSF2 : TYPE
        DESCRIPTION.
    display : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    0 -> identical images, 1-> image compared to nothing, inf -> image and its negative

    """
    
    if PSF1.shape != PSF2.shape :
        raise ValueError(f'The two PSF have different shapes {PSF1.shape} and {PSF2.shape}')
    
    dif = PSF1/np.sum(PSF1) - PSF2/np.sum(PSF2)
    
    if display:
        plt.figure()
        plt.imshow(np.log(dif))
        plt.colorbar()
        plt.title("PSFsimlarity")
        plt.show()
    
    return np.std(dif)
    

def showKL(index, M2C):
    if index >= M2C.shape[1]:
        raise ValueError("Index out of range")
    dm.coefs = 0
    dm.coefs = M2C[:, index] * 5e-6
    tel*dm
    tel.computePSF(4)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(tel.OPD)
    plt.title(f"OPD for KL n°{index}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    L = tel.PSF.shape[0]
    d = 80
    plt.imshow(tel.PSF[d:L-d, d:L-d])
    plt.title(f"PSF for KL n°{index}")
    plt.axis('off')
    plt.show()








