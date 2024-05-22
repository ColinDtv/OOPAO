# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:46:13 2024

@author: cdartevelle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def LIFT(Telescope, DeformableMiror, Camera, Source,
         phaseDiversity, PSF_Diversity, M2C_KL,
         RMSamplitude:float=10e-9, zeroPaddingFactor:float=2,
         nIterrationsMax:int=20, nIterrationsMin:int=5, nModes:int=15,
         ConvergenceThreshold:float=1e-10, loopGain:float=1, Display:bool=False):
    
    """   =======================🏋️ LIFT help 🏋️‍==========================
    
       | LIFT : estimation of the modes of a PSF, using the phase diversity method
       |
    Parameters
    ----------
       | Telescope : OOPAO object
       | DeformableMiror : OOPAO object
       | Camera : OOPAO object
       | Source : OOPAO object
       | 
       | phaseDiversity : known aberration applied to generate the PSF.
       |     Can be given as haseDiversity = M2C[:,mode] * amplitude
       | PSF_Diversity : PSF to be analysed, in form of a numpy array
       | M2C_KL : Matrix for conversion of the modes to dm commands
       | 
       | RMSamplitude : Ooptional, the default is 10e-9 [m]
       |     Amplitude used in the push-pull method to generate the interaction matrix
       | zeroPaddingFactor : optional, the default is 2.
       |     factor for zeroPadding used by the function tel.computePSF() 
       | nIterrationsMax : optional, the default is 20.
       |     maximum number of itterations done by the function before stoping computation
       | nIterrationsMin : optional, the default is 5.
       |     minimum number of itterations done by the function before stoping computation
       |      - set both nIterrationsMin and ConvergenceThreshold to None to run
       |        until nIterrationsMax
       | nModes : optional, the default is 15.
       |     number of modes from the M2C_KL matris for the reconstruction
       | ConvergenceThreshold : optional, the default is 1e-10 [m].
       |     minimum variation from one iteration to the folowing to stop computation
       |      - set both nIterrationsMin and ConvergenceThreshold to None to run
       |        until nIterrationsMax
       | loopGain : optional, the defualt is 1.
       |     Intergartor gain for the loop
       | Display : optional, the default is False.
       |     display the estimation at each itteration with a graph if True
       |
    Returns
    -------
       | converged : boolean
       |     True if the convergence criterion was reached at the end of execution,
       |     False if not
       | estimation : list of the coefficients of the modes of the PSF (lenght = nModes)
       |     In case LIFT did not converge, returns a list of np.nan
       |     The dm command to recreate the estimated PSF is M2C[:, :nModes] @ estimation
       |
       =======================🏋️ LIFT help 🏋️‍==========================
    """
    
    #selecting the used modes for the M2C matrix
    M2C = M2C_KL[:, :nModes]
    
    #getting the size of the generated PSFs
    Telescope.resetOPD()
    DeformableMiror.coefs = 0
    Telescope*DeformableMiror
    Telescope.computePSF(zeroPaddingFactor)
    [m, n] = np.shape(Telescope.PSF)
    
    #initialising variables
    IM = np.zeros([m*n, nModes])   #empty interaction matrix
    PSF0 = np.zeros([m, n])        #empty PSF matrix
    estimation = np.zeros(nModes)  #empty estimation vetor
    estim_history = []             #vector of every estimations
    PSF0 = []                      #vector of every PSF computed
    PSF1 = PSF_Diversity.ravel()   #linearized version of PSF_Diversity
    gap = []                       #gap between on estimation and the next one
    end = False                    #stop loop when convergence
    k = 0                          #itteration counter
    
    if nIterrationsMin == None :
        nIterrationsMin = nIterrationsMax
    
    while not end :
        #computation of the reconstruction matrix with the Push-Pull method
        for i in range(nModes):
            Zi = M2C[:, i] * RMSamplitude
            
            #push
            DeformableMiror.coefs = 0
            DeformableMiror.coefs = phaseDiversity + Zi + M2C @ estimation
            Telescope*DeformableMiror
            Telescope.computePSF(zeroPaddingFactor)
            Spush = Telescope.PSF
            
            #pull
            DeformableMiror.coefs = 0
            DeformableMiror.coefs = phaseDiversity - Zi + M2C @ estimation 
            Telescope*DeformableMiror
            Telescope.computePSF(zeroPaddingFactor)
            Spull = Telescope.PSF
            
            #interaction matrix for the mode
            IMzi = (Spush - Spull) / (2 * RMSamplitude)
            IM[:, i] = np.ravel(IMzi)
        
        reconstructor = np.linalg.pinv(IM)
        
        #computation of the new candidate PSF
        DeformableMiror.coefs = phaseDiversity + M2C @ estimation 
        Telescope*DeformableMiror
        Telescope.computePSF(zeroPaddingFactor)
        PSF0.append(Telescope.PSF.ravel())
        
        #computation of the new estimation
        estim_old = estimation.copy()
        estimation = np.matmul(reconstructor, PSF1 - PSF0[-1]) * loopGain + estim_old
        estim_history.append(estimation)
        
        #difference between the current estimation and the previous one
        gap.append(gapEstimations(estimation, estim_old))
        
        #progress display
        if Display :
            alpao_unit = 7591.024876 * 1e-9
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.grid()
            plt.bar(range(nModes), estimation/alpao_unit)
            plt.title(f"LIFT estimation for {k} iterrations")
            plt.xlabel("KL mode index")
            plt.ylabel("Mode amplitude [DM_unit]")
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
            plt.subplot(1, 2, 2)
            plt.plot(gap, '-+', label="{:.2e}".format(gap[-1]))
            if ConvergenceThreshold != None :
                plt.plot(np.ones(k+1)*(gap[-1]+ConvergenceThreshold), '--', label='threshold', color='orange')
                plt.plot(np.ones(k+1)*(gap[-1]-ConvergenceThreshold), '--', color='orange')
                plt.fill_between(range(k+1),np.ones(k+1)*(gap[-1]+ConvergenceThreshold),np.ones(k+1)*(gap[-1]-ConvergenceThreshold), color='orange', alpha=0.2)
            plt.title("Difference between two last estimations")
            plt.xlabel("Iterration")
            plt.ylabel("Difference [m]")
            plt.legend()
            
            plt.tight_layout(pad=0.5)
            plt.pause(0.01)
            
        #testing convergence :
        if k >= nIterrationsMin and ConvergenceThreshold != None:
            if convergence(gap, k, 5, ConvergenceThreshold):
                end = True
                converged = True
                print(f"LIFT sucesfully converged after {k} itterations")
                
                #results dispaly
                if Display :
                    plt.figure()
                    plt.imshow(Telescope.PSF)
                    plt.title(f"Last PSF estimated\n({k} itterations)")
            
        #updating loop counter :
        k += 1
        
        #if max number of ittretaions exceeded, loop is stopped
        if k >= nIterrationsMax :
            end = True
            converged = False
            print(f"LIFT is unable to converge after {k} itterations")        
            
    return converged, estimation




def convergence(gap, current_iter:int, check_lenght:int, epsilon:float):
    
    if check_lenght < 2:
        raise ValueError(f"check_lenght must be grater than 2 ({check_lenght} was given)")
    
    if current_iter >= check_lenght:
        convergence = True
        for i in range(2, check_lenght):
            if np.abs(gap[-1] - gap[-i]) >= epsilon :
                convergence = False
        return convergence
    
    return False



def gapEstimations(estim1, estim2):
    if len(estim1) != len(estim2):
        raise ValueError(f"The two estimations are different sizes {len(estim1)} and {len(estim2)}")
    #cor = np.correlate(estim1, estim2)
    ecart = np.std(estim1-estim2)
    return ecart



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



def cogPSF(PSF, search_radius:float=5):
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
    
    center = np.zeros(PSF.shape)
    r = search_radius
    center[maxIx-r:maxIx+r, maxIy-r:maxIy+r] = PSF[maxIx-r:maxIx+r, maxIy-r:maxIy+r]

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



def LIFTwelcome():
    """Just type \'LIFTwelcome()\' !"""
    print('\n')
    print('======================🏋️ Welcome to LIFT 🏋️‍=======================')
    print('\n')
    print('          ////          ////   ////////////  ////////////////     ')
    print('         ////          ////   ////////////  ////////////////      ')
    print('        ////                 ////                ////             ')
    print('       ////          ////   ////                ////   /\         ')
    print('      ////          ////   ////                ////   /  \     ')
    print('     ////----------////---////////------------////---/    \  ')
    print('    ////----------////---////////------------////---/      \  ')
    print('   ////          ////   ////                ////   /    _.=^      ')
    print('  ////////////  ////   ////                ////   /_.=^          ')
    print(' ////////////  ////   ////                ////                    ')
    print('\n')
    print('=============🏋️ type \'help(LIFT)\' for more infos 🏋️‍===============')
    print('\n')



if __name__ != '__main__':
    LIFTwelcome()


#EOF