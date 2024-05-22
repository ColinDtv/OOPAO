# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:32:35 2024

@author: cdartevelle
"""

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
from codes_perso.Outils import *


#%% lecture des fichiers .npy
folder = r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\Données PAPYRUS (mission OHP 26-27-28 mars)\m_kl_4"

files = glob.glob(os.path.join(folder, '*.npy'))
nb_files = len(files)
print(f"{nb_files} .npy files sucessfully imported :")
file_names = [os.path.basename(file) for file in files]

# mat = loadmat(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\Données PAPYRUS (mission OHP 26-27-28 mars)\M2C_KL_OOPAO_synthetic_IF.mat")
# M2C_papy = mat['M2C_KL']

mat = loadmat(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\M2C_KL_OOPAO_synthetic_IF.mat")
M2C_mat = mat['M2C_KL']

#V2Z = np.load(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\V2Z_PAPYRUS.npy")

dark = np.load(r"C:\Users\cdartevelle\Documents\Banc PAPYRUS\Données PAPYRUS (mission OHP 26-27-28 mars)\dark.npy")

fig, axs = plt.subplots(5, 4, figsize=(12, 8))
axs = axs.flatten()

for i in range(nb_files):
    print(i, ' - ', file_names[i])
    data = np.load(files[i])
    axs[i].imshow(data)
    axs[i].set_title(file_names[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

#%% import paramètres PAPYRUS
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
    

#%%
i = 11     #10
print(f"Processing file {i}/{nb_files-1} : {file_names[i]}")
PSF = np.load(files[i])
# TF = ft2(PSF)
M2C = M2C_mat
# TF[256, :] = 0
# TF[:, 320] = 0

param['rotationAngle'] = 90
misReg = MisRegistration(param)
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

zpad = 6
Phase_amp = path2amp(file_names[i])
Div_amp = 0.1
Phase_mode = 3
DMunit = 7.5e-6#10e-6 #7591e-9

size = tel.resolution*zpad
filePSF = centerPSF(PSF - dark, (size, size))
filePSF = (filePSF > 0)*filePSF

#filePSF = np.transpose(filePSF)
#filePSF = (filePSF*np.sum(telPSF))/np.sum(filePSF)

#normalisation du flux
Flux_tot = filePSF.sum()
Surf_tot = tel.pixelArea * (tel.D/tel.resolution)**2
nPhot = Flux_tot / (Surf_tot * tel.samplingTime)
#ngs.nPhoton = nPhot


#PSF simulée
dm.coefs = (Phase_amp*M2C[:, Phase_mode] + Div_amp*M2C[:, 2]) * DMunit
tel.resetOPD
cam.photonNoise = False
cam.readoutNoise = False   #for CRED-2 [e-] (<30e-)
#ngs.nPhoton = 2000000
ngs*tel*dm
tel*cam
tel.computePSF(zpad)
telPSF = tel.PSF#cam.frame#tel.PSF


#affichage
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(telPSF, cmap='gist_ncar')
# plt.title("PSF simulée")
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(filePSF, cmap='gist_ncar')
# #plt.colorbar()
# plt.title("PSF réelle")
# plt.colorbar()
# plt.show()

#affichage concaténé (même échelle de couleurs)
disp = np.zeros([size, size*2])
disp[:, :size] = telPSF
disp[:, size:] = filePSF
plt.figure()
plt.imshow(disp, cmap='gist_ncar')
plt.title("PSF simulée // PSF réelle")
plt.colorbar()
plt.show()

print(f"écart entre les PSF : {PSFdisimilarity(telPSF, filePSF)}")

#%%

PSF = filePSF
#PD = M2C[:, 2] * 150 * 1e-9
thresh = 1e-8
gain = 2

DM_amplitude = 2.66e-5 #conversion DM-unit / [m]
#PD = dm.modes @ M2C[:,2] * ampRMS
PD = M2C[:,2] * Div_amp * DMunit

#PD = M2C[:,2] * V2Z[2] * 0.2

# PD = [vect coefs modaux] @ M2C @ M2Φ
#avec matrice c2P à recréer avec les fonctions d'influence (de dim nombre de pixels du détecteur x nombre d'actuateurs)
#fixhiers data V2Z (.npy) sur NC (coefs DM vers modes de Zernike)


tel.computePSF(zpad)
#cropPSF = crop(PSF, tel.PSF.shape)
plt.figure()
plt.imshow(np.log(PSF))
plt.colorbar()
plt.show()


conv, estim = LIFT(Telescope=tel, DeformableMiror=dm, Camera=cam, Source=ngs, phaseDiversity=PD,
             PSF_Diversity=PSF, M2C_KL=M2C[:,0:], zeroPaddingFactor=zpad, nIterrationsMax=50,
             nIterrationsMin=30, nModes=10, ConvergenceThreshold=thresh, loopGain=gain, Display=True)


#%% comparaison PSF


dm.coefs = 0
dm.coefs = M2C[:, :len(estim)] @ estim + PD
tel.resetOPD()
ngs*tel*dm
tel.computePSF(zpad)
estPSF = tel.PSF


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(filePSF)
plt.title("PSF réelle")
plt.subplot(1, 2, 2)
plt.imshow(estPSF)
plt.title("PSF estimée")

print(f"écart entre les PSF : {PSFdisimilarity(filePSF, estPSF)}")






#%% Influence de l'inclinaison

thresh = 1e-8
nModes_div = 10
zpad = 4
PD = 200*M2C[:,2] * 1e-9
dm.coefs = 0
dm.coefs = M2C[:, 7] * 50e-9 + PD
tel.resetOPD()
ngs*tel*dm
tel.computePSF(zpad)

telPSF = tel.PSF

param['rotationAngle'] = 0
misReg = MisRegistration(param)
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)


# PD = 0.2*M2C[:,2] * 1e-9
# dm.coefs = 0
# dm.coefs = M2C[:, 8] * 1e-9 + PD
# tel.resetOPD()
# ngs*tel*dm
# tel.computePSF(zpad)
# PSF = tel.PSF


angles = []
ecarts = []

fig, axs = plt.subplots(5, 4, figsize=(12, 8))
axs = axs.flatten()

for i in range(-10, 10):
    angles.append(i/2)
    print(f"angle de {angles[-1]}°")
    
    param['rotationAngle'] = angles[-1]
    misReg = MisRegistration(param)
    dm = DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg)
    
    conv, estim = LIFT(Telescope=tel, DeformableMiror=dm, Camera=cam, Source=ngs, phaseDiversity=PD,
                 PSF_Diversity=telPSF, M2C_KL=M2C[:,0:], zeroPaddingFactor=zpad, 
                 nIterrationsMax=20, nModes=10, ConvergenceThreshold=thresh, loopGain = 1, Display=False)
    
    dm.coefs = M2C[:, :len(estim)] @ estim + PD
    tel.resetOPD()
    ngs*tel*dm
    tel.computePSF(zpad)
    estPSF = tel.PSF
    
    
    axs[i].imshow(filePSF)
    axs[i].set_title(f"angle = {angles[-1]}°")
    
    ecarts.append(PSFdisimilarity(estPSF, telPSF))

plt.tight_layout(pad=15)
plt.show()

plt.figure()
plt.plot(angles, ecarts, '-+')
plt.legend()
plt.xlabel("Angle du dm (°)")
plt.ylabel("Ecart entre les PSF réelle est estimée")
plt.title("Estimations LIFT pour différentes orientations du dm")





#%% influence du sampling

zpad = 4
zpads = np.arange(zpad-0.75, zpad+1, 0.25)
ecarts = [0] * zpads.shape[0]

PD_coefs = np.asarray([0, 0, 50, 0, 0, 0, 0, 0, 0, 0]) * 1e-9
PD = M2C[:,:10] @ PD_coefs
PA_coefs = np.asarray([0, 0, 0, 0, 0, 0, 0, 50, 0, 0]) * 1e-9 + PD_coefs
PA = M2C[:,:10] @ PA_coefs

dm.coefs = 0
dm.coefs = PA
ngs*tel*dm
tel.computePSF(zpad)
refPSF = tel.PSF.copy()
refPhase = dm.OPD*tel.pupil

fig, axs = plt.subplots(4, 4, figsize=(12, 8))
axs = axs.flatten()
k = 0

for i in range(len(zpads)):
    print('======================')
    print('  zeropad = ', zpads[i])
    
    dm.coefs = 0
    ngs*tel*dm
    tel.computePSF(zpads[i])
    size = tel.PSF.shape
    cropPSF = centerPSF(refPSF, size)
    print('  taille = ', size)
    
    conv, estim = LIFT(Telescope=tel, DeformableMiror=dm, Camera=cam, Source=ngs,
                 phaseDiversity=PD, PSF_Diversity=cropPSF, M2C_KL=M2C,
                 zeroPaddingFactor=zpads[i], nIterrationsMax=40, nIterrationsMin=20,
                 nModes=10, ConvergenceThreshold=1, loopGain = 0.5, Display=False)
    
    dm.coefs = 0
    dm.coefs = M2C[:, :len(estim)] @ estim + PD
    ngs*tel*dm
    tel.computePSF(zpads[i])
    estPSF = tel.PSF.copy()
    estPhase = dm.OPD*tel.pupil
    
    ecarts[i] = np.nanstd(refPhase - estPhase)
    print('  écart  =', ecarts[i])
    
    axs[k].imshow(centerPSF(cropPSF, (40, 40)))
    axs[k].set_title(f"input s = {zpads[i]}")
    k+=1
    axs[k].imshow(centerPSF(estPSF, (40, 40)))
    axs[k].set_title(f"estim s = {zpads[i]}")
    k+=1

fig.tight_layout(pad=15)
fig.show()

plt.figure()
plt.plot(zpads, ecarts, '-+')
plt.legend()
plt.yscale('log')
plt.xlabel("Echantillonnage")
plt.ylabel("Ecart RSM entre les phases d'entrée et estimés")
plt.title("Estimations LIFT pour différents échantillonnages")
plt.show()

    
    
    




#%%


vrai_sampling = 4
faux_sampling = 2

PSF_originale = tel.computePSF(vrai_sampling)
Fausse_PSF = recadre(PSF_originale, size(tel.computePSF(faux_sampling)))

estimation = LIFT(fausse_PSF, faux_sampling)
PSF_estimée = générer_PSF(estimation)

écart = différence (Fausse_PSF, PSF_estimée)
















