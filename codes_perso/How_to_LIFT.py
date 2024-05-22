# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:09:33 2024

@author: cdartevelle
"""

#%% Import librairies

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import sys
from scipy.io import loadmat

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.DetectorV2 import Detector

from codes_perso.LIFT import *



#%% create the OOPAO objects, even for processing real data

from parameter_files.parameterFile_simple_Papyrus import initializeParameterFile

param = initializeParameterFile()

plt.ion()

def printf(line):
    sys.stdout.write(line)
    sys.stdout.flush()

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

# create the Source object
ngs=Source(optBand            = param['opticalBand'],\
           magnitude          = param['magnitude'],\
           display_properties = False)

ngs*tel

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)
atm.update()
tel+atm

# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)
tel*dm

# make sure tel and atm are separated to initialize the PWFS
tel-atm

wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = param['n_pix_separation'],\
              psfCentering          = True,\
              postProcessing        = 'slopesMaps_incidence_flux')    

M2C_KL = compute_KL_basis(tel, atm, dm, lim=1e-2)
stroke=1e-9

calib = InteractionMatrix(ngs            = ngs,
                          atm            = atm,
                          tel            = tel,
                          dm             = dm,
                          wfs            = wfs,
                          M2C            = M2C_KL,
                          stroke         = stroke,
                          nMeasurements  = 20,
                          noise          = 'off',
                          print_time     = False)

cam = Detector(nRes = tel.resolution)
cam.maximum = 1
cam.psf_sampling = 2

def center(image):
    image = image > image.mean()*2
    
    rows, cols = np.indices(image.shape)
    Itot = np.sum(image)
    x = np.sum(cols * image) / Itot
    y = np.sum(rows * image) / Itot
    return (x, y)



def crop(image, size):
    (x, y) = center(image)
    x = int(x)
    y = int(y)
    (X, Y) = size
    if x < X or y < Y:
        raise ValueError("PSF is too offcentered to be cropped")
    return image[y-Y//2:y+Y//2, x-X//2:x+X//2]


def path2amp(file_name):
    dot = int(file_name[-5])
    unit = int(file_name[-7])
    sign = file_name[-8]
    
    if sign=='p':
        amp = unit + 0.1*dot
    elif sign=='m':
        amp = -unit -0.1*dot
    return amp



#%% Import your files to process

#M2C matrix from the bench
mat = loadmat(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\M2C_KL_OOPAO_synthetic_IF.mat")
M2C = mat['M2C_KL']

#the dark image for noise
dark = np.load(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\Données PAPYRUS (mission OHP 26-27-28 mars)\dark.npy")

#PSF to process
PSF = np.load(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\Données PAPYRUS (mission OHP 26-27-28 mars)\m_kl_4\m0p2.npy")



#%% Pre-process the PSF

#get a reference OOPAO PSF
ZeroPadding = 4            #need to match the one on the file
dm.coefs = 0
tel*dm
tel.computePSF(ZeroPadding)

#crop and center the PSF
PSF = PSF - dark
cleanPSF = centerPSF(PSF, tel.PSF.shape)

plt.figure()
plt.imshow(cleanPSF)

#set the diversity used
Diversity = M2C[:,2] * 1e-9   #need to match the one on the file



#%% Do the LIFT

nModes = 10 #number of modes used for reconstruction
#gap between iterations to dermine the convergence of algorithm, high value = low precision but better convergence :
ConvergenceThreshold = 1e-9
Display = False #set to true to see graph of the evolution of the algorithm


estimation = LIFT(Telescope=tel, DeformableMiror=dm, Camera=cam, Source=ngs, phaseDiversity=Diversity,
             PSF_Diversity=cleanPSF, M2C_KL=M2C[:,:], zeroPaddingFactor=ZeroPadding, nIterrationsMax=50,
             nIterrationsMin=5, nModes=nModes, ConvergenceThreshold=ConvergenceThreshold, Display=Display)

plt.figure()
plt.plot(estimation, '-+')
plt.xlabel("KL mode index")
plt.ylabel("Mode amplitude (m)")
plt.title("LIFT estimation")
plt.show()



#%% visualise the PSF guessed by LIFT

dm.coefs = 0
dm.coefs = M2C[:, :len(estimation)] @ estimation + Diversity
tel.resetOPD()
ngs*tel*dm
tel.computePSF(ZeroPadding)
estimated_PSF = tel.PSF

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cleanPSF)
plt.title("Input PSF")
plt.subplot(1, 2, 2)
plt.imshow(estimated_PSF)
plt.title("Estimated PSF")
plt.show()












