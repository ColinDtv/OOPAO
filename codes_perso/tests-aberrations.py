# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:35:03 2024

@author: cdartevelle
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import filters

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
from OOPAO.Pyramid import Pyramid


#%% télescope

n_subaperture = 40
res = 10*n_subaperture
diam = 10

tel = Telescope(resolution=res, diameter=diam, samplingTime=1/1000, centralObstruction=0.2, display_optical_path=True)

thickness_spider    = 0.05
angle               = [15, 165, 195, 345]
offset_Y            = None
offset_X            = None

tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

plt.figure()
plt.imshow(tel.pupil, "gray")

#%% source

source_WFS = Source(optBand='I', magnitude=4) #source pour le WFS
source_sci = Source(optBand='K', magnitude=4) #source pour les instruments scientifiques

source_WFS*tel

tel.src.print_properties()

tel.computePSF(zeroPaddingFactor = 6)
PSF = np.abs(tel.PSF)

plt.figure()
plt.imshow(np.log10(PSF))
plt.clim([-3, 4])
plt.colorbar()


#%% calcul PSF manuel

#image de la source
pad = 5  #facteur de 0-padding
flux = 2e9 #phot/s/m²

# champ electro-magnetique
offset = res*pad//2 - res//2
psi = np.zeros([res*pad, res*pad], dtype=np.complex128)
psi[offset:offset+res, offset:offset+res] = tel.pupil * flux * np.exp(1j*tel.pupil)

#convolution
psi_TF = np.fft.fftshift(np.fft.fft2(psi))
PSF = np.abs(psi_TF)**2
PSF /= PSF.max()

plt.figure()
plt.imshow(np.log10(PSF))

plt.clim([-6, 0])
plt.title("PSF manuelle")
plt.show()

#%% atmosphère

r0 = 0.5
L0 = 10
fractionalR0 = [0.5, 0.1, 0.8]
windSpeed = [5, 20, 15]
windDirection = [0, 45, 180]
altitude = [0, 2000, 10000]

atm = Atmosphere(tel, r0, L0, windSpeed, fractionalR0, windDirection, altitude) 
atm.initializeAtmosphere(tel)
#atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

atm.display_atm_layers()

#%% Effet atmosphère manuel

# champ electro-magnetique
offset = res*pad//2 - res//2
psi = np.zeros([res*pad, res*pad], dtype=np.complex128)
psi[offset:offset+res, offset:offset+res] = atm.OPD * flux * np.exp(1j*atm.OPD)

#convolution
psi_TF = np.fft.fftshift(np.fft.fft2(psi))
PSF = np.abs(psi_TF)**2
PSF /= PSF.max()

plt.figure()
plt.imshow(np.log10(PSF))
plt.clim([-6, 0])
plt.title("PSF avec atm manuelle")
plt.show()

#%% boucle atm
"""
plt.figure()

n = 20
for i in range(n):
    atm.update()

    plt.imshow(atm.OPD*1e9)
    plt.title(f"{i+1}/{n}")
    plt.pause(0.01)"""

#%% création DM

misReg = MisRegistration()
misReg.rotationAngle = 0
mechCoupling = 0.35
nAct = 13
pitch = tel.D/nAct

dm = DeformableMirror(telescope=tel, nSubap=nAct-1, mechCoupling=mechCoupling, misReg=misReg, coordinates=None, pitch=pitch)


plt.figure()
plt.imshow(tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'+')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('Position actuateurs DM')


#%% propagation

tel.resetOPD()
source_WFS*tel
tel+atm

tel.computePSF(4)
PSF = np.abs(tel.PSF)

telPSF = np.log10(PSF)

plt.figure()
plt.imshow(telPSF)
plt.clim([-3, 4])
plt.colorbar()
plt.title("PSF après atmosphère")


#%% Pilotage du DM

tel.resetOPD()

n = dm.nValidAct
dm.coefs = np.zeros(n)
dm.coefs[:n//4] = 1e-9
dm.coefs[n//4 : n//2] = -1e-9
dm.coefs[n//2 : 3*n//4] = 1e-9
dm.coefs[3*n//4:] = 1e-9
dm.coefs = dm.coefs

plt.figure()
plt.imshow(dm.OPD)
plt.colorbar()
plt.title("OPD DM")
plt.show()

tel.computePSF(4)
PSF = np.abs(tel.PSF)
telPSF = np.log10(PSF)

source_WFS*tel*dm

tel.computePSF(4)
PSF = np.abs(tel.PSF)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(telPSF)
plt.clim([-3, 4])
plt.colorbar()
plt.title("PSF avant DM")
plt.subplot(1, 2, 2)
plt.imshow(np.log10(PSF))
plt.clim([-3, 4])
plt.colorbar()
plt.title("PSF après DM")
plt.show()


#%% Mise en évidence diversité de phase
mode = 2 #mode utilisé, de 1 à 10
modes = 100

tel.resetOPD()
Z = Zernike(tel, modes)
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

PD = M2C_zernike[:, 2]*5e-7
dm.coefs = 0
dm.coefs = -M2C_zernike[:, mode-1]*1e-6 + PD

tel.computePSF(4)
PSF = np.abs(tel.PSF)
telPSF = np.log10(PSF)

source_WFS*tel*dm

tel.computePSF(4)
PSF = np.abs(tel.PSF)

#affichage

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(dm.OPD)
plt.title(f"OPD mode de Zernike n°{mode}")

plt.subplot(1, 3, 2)
plt.imshow(telPSF)
plt.clim([-3, 4])
plt.title("PSF avant DM")

plt.subplot(1, 3, 3)
plt.imshow(np.log10(PSF))
plt.clim([-3, 4])
plt.title("PSF après DM")

plt.tight_layout(pad=1)
plt.show()

#%% Extraction de la phase (diversité de phase)

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

wfs = ShackHartmann(nSubap=n_subaperture, telescope=tel, lightRatio=0.5,
                    binning_factor=1, is_geometric=False, shannon_sampling=True)

#tel*wfs
source_WFS*tel*dm*wfs
tel+atm

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('Pixels du WFS')

plt.figure()
plt.imshow(wfs.signal_2D)
plt.title("Signal mesuré par le WFS")


#%%

tel.isPaired = False
tel.resetOPD()
wfs.is_geometric = True

#matrice d'interraction
M2C_modal = M2C_zernike
stroke=1e-9
calib_zernike = InteractionMatrix(ngs=source_WFS, atm=atm, tel=tel, dm=dm, wfs=wfs,
                                M2C=M2C_modal, stroke=stroke, nMeasurements=100, noise='off')
tel.isPaired = False
tel.resetOPD()
wfs.is_geometric = False
#%% extraction de la phase SH
reconstructor = M2C_zernike @ calib_zernike.M

dm.coefs = M2C_zernike@np.random.randn(modes) * 100e-9
tel-atm

tel*dm*wfs

phase_introduite = tel.src.phase.copy()

plt.figure()
plt.subplot(1,3,1)

plt.imshow(tel.src.phase)
plt.title("Phase Introduite")
plt.colorbar()
# plt.clim([-1.75,1.75])


dm_commands = np.matmul(reconstructor, wfs.signal)

dm.coefs = dm_commands


tel*dm

plt.subplot(1,3,2)
plt.imshow(tel.src.phase)
plt.title("Phase Reconstruite")
plt.colorbar()
# plt.clim([-1.75,1.75])


phase_reconstruite = tel.src.phase.copy()
plt.subplot(1,3,3)

plt.imshow(phase_introduite-phase_reconstruite)
plt.title("Phase Résiduelle")
# plt.clim([-1.75,1.75])
plt.colorbar()


#%% WFS Pyramide

modulation = 0.5 #[λ/D]
lightRatio = 0.8
tel.resetOPD()

pwfs = Pyramid(nSubap=n_subaperture, telescope=tel, modulation=modulation, lightRatio=lightRatio)

source_WFS*tel*dm*pwfs
#tel+atm

plt.figure()
plt.imshow(pwfs.cam.frame)
plt.title('Pixels du WFS')

plt.figure()
plt.imshow(pwfs.signal_2D)
plt.title("Signal mesuré par le WFS")

#%% Extraction de la phase Py



calib_zernike_py = InteractionMatrix(ngs=source_WFS, atm=atm, tel=tel, dm=dm, wfs=pwfs,
                                M2C=M2C_modal, stroke=stroke, nMeasurements=100, noise='off')

reconstructor = M2C_zernike @ calib_zernike_py.M
dm.coefs = M2C_zernike@np.random.randn(modes) * 50e-9

tel-atm
tel*dm*pwfs
phase_introduite = tel.src.phase.copy()


#limites d'affichage :
lim_min = -10
lim_max = 10

plt.figure()
plt.subplot(1,3,1)
plt.imshow(tel.src.phase)
plt.title("Phase Introduite")
plt.colorbar()
plt.clim([lim_min, lim_max])

dm_commands = np.matmul(reconstructor, pwfs.signal)
dm.coefs = dm_commands
tel*dm

plt.subplot(1,3,2)
plt.imshow(tel.src.phase)
plt.title("Phase Reconstruite")
plt.colorbar()
plt.clim([lim_min, lim_max])

phase_reconstruite = tel.src.phase.copy()
plt.subplot(1,3,3)

plt.imshow(phase_introduite-phase_reconstruite)
plt.title("Phase Résiduelle")
plt.clim([lim_min, lim_max])
plt.colorbar()












