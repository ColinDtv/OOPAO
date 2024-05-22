# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:45:37 2024

@author: cdartevelle
"""

import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector
from OOPAO.LiFT import LiFT
from OOPAO.tools.tools import crop

import matplotlib.ticker as ticker
from scipy.io import loadmat

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

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
                samplingTime        = 5e-3,\
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
    



#%%

mat = loadmat(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\M2C_KL_OOPAO_synthetic_IF.mat")
M2C_mat = mat['M2C_KL']

dm.coefs = 0
dm.coefs = M2C_mat
ngs*tel*dm
KL_basis = tel.OPD


det = Detector(readoutNoise=4,photonNoise=True)

det.integrationTime=1*tel.samplingTime

ngs*tel*det

plt.imshow(crop(tel.PSF,20,axis=0))
plt.colorbar()
plt.show()

det = Detector(readoutNoise=4,photonNoise=True)

det.integrationTime=2*tel.samplingTime

ngs*tel*det

plt.imshow(crop(tel.PSF,20,axis=0))
plt.colorbar()
plt.show()


#%%
alpao_unit = 7591.024876 * 1e-9
zpad = 3.7

nModes = 50
Div_mode = 2
Div_amp = 0.2
Ab_mode = 3
Ab_amp = 0

amp_diversity = np.zeros(nModes)
amp_diversity[Div_mode] = Div_amp*alpao_unit

diversity_OPD = KL_basis[:, :, :nModes]@amp_diversity

plt.imshow(diversity_OPD)
plt.title("OPD diversity")
plt.colorbar()
plt.show()

dm.coefs = 0
dm.coefs = (Ab_amp*M2C_mat[:, Ab_mode] + Div_amp*M2C_mat[:, Div_mode]) * alpao_unit
ngs*tel*dm
tel.computePSF(zpad)
telPSF = tel.PSF

plt.figure()
plt.imshow(telPSF)
plt.show()

ang_pixel_rad = tel.src.wavelength / (zpad*tel.D)
ang_pixel = int((ang_pixel_rad/np.pi) * 180*3600*1000)

estimator = LiFT(tel = tel, basis = KL_basis, diversity_OPD = diversity_OPD,iterations = 20, det = det, ang_pixel = ang_pixel, img_resolution = tel.resolution*zpad)

modes = [0,1,2,3,4,5,6,7,8,9]
coefs_1, PSF_1, _ = estimator.Reconstruct(telPSF, R_n='model', mode_ids=modes)

plt.figure()
plt.grid()
plt.bar(range(10), coefs_1[:10]/alpao_unit)
plt.title(f"LIFT estimation")
plt.xlabel("KL mode index")
plt.ylabel("Mode amplitude [DM_unit]")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
