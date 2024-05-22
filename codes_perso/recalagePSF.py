# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:58:03 2024

@author: cdartevelle
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
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
from OOPAO.Zernike import Zernike

from codes_perso.LIFT import *
from codes_perso.Outils import *


#%% import paramètres PAPYRUS
from parameter_files.parameterFile_simple_Papyrus import initializeParameterFile

param = initializeParameterFile()

plt.ion()

def printf(line):
    pass
    #sys.stdout.write(line)
    #sys.stdout.flush()

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
param['rotationAngle'] = -90
misReg = MisRegistration(param)
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

param['rotationAngle'] = -90.69
misReg = MisRegistration(param)
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)
    
Z = Zernike(tel,195)
Z.computeZernike(tel)
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

#%% angle du miroir

mode = 9

param['rotationAngle'] = 0
misReg = MisRegistration(param)
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

dm.coefs = 0
dm.coefs = M2C_zernike[:, mode] * 1e-7
tel*dm
tel.computePSF()
refPSF = tel.PSF.copy()    

angles = []
ecarts = []

for i in range(-30, 30):
    angles.append(i/10)
    
    param['rotationAngle'] = angles[-1]
    misReg = MisRegistration(param)
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg)

    dm.coefs = 0
    dm.coefs = M2C_zernike[:, mode] * 1e-7
    tel*dm
    tel.computePSF()
    
    ecarts.append(PSFdisimilarity(refPSF, tel.PSF))


plt.figure()
plt.plot(angles, ecarts, '-+')
plt.title(f"Ecart entre PSF en fonction de l'orientation pour le mode de Zernike {mode}")
plt.xlabel("Différence d\'angle (°)")
plt.ylabel("Ecart entre les PSF (m RMS)")
plt.show()


#%% Echantillonage

mode = 3

param['rotationAngle'] = 0
misReg = MisRegistration(param)
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

zpad = 4
dm.coefs = 0
dm.coefs = M2C_zernike[:, mode] * 1e-7
tel*dm
tel.computePSF(zpad)
size = zpad*tel.resolution
refPSF = tel.PSF.copy()    

echant = []
ecarts = []

for i in range(-80, 80):
    #print("i =", i)

    dm.coefs = 0
    dm.coefs = M2C_zernike[:, mode] * 1e-7
    tel*dm
    tel.computePSF(zpad + i/80)
    telPSF = tel.PSF

    L = telPSF.shape[0]
    if L%2 == 0:
        echant.append(i/80)
        d = int((L-size)/2)
    
        if i>0: #recadrage
            start_row = (L - size) // 2
            end_row = start_row + size
            
            PSF = telPSF[start_row:end_row, start_row:end_row]
            
        if i<0: #padding
            pad_top = (size-L) // 2
            pad_bottom = (size-L) - pad_top
    
            PSF = np.pad(telPSF, ((pad_top, pad_bottom), (pad_top, pad_bottom)), mode='constant')
    
        ecarts.append(PSFdisimilarity(refPSF, PSF))


plt.figure()
plt.plot(echant, ecarts, '-+')
plt.title(f"Ecart entre PSF en fonction de l'échantillonage")
plt.xlabel("Différence d\'échnatillonage")
plt.ylabel("Ecart entre les PSF (m RMS)")
plt.show()



