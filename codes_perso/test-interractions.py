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
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis


from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
from OOPAO.Pyramid import Pyramid


#%% télescope

n_subaperture = 20
res = 6*n_subaperture
diam = 10

tel = Telescope(resolution=res, diameter=diam, samplingTime=1/1000, centralObstruction=0.2, display_optical_path=True)

thickness_spider    = 0.05
angle               = [45, 135, 225, 315]
offset_Y            = None
offset_X            = None

tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

plt.figure()
plt.imshow(tel.pupil, "gray")

#%% source

ngs = Source(optBand='I', magnitude=4) #source pour le WFS
source_sci = Source(optBand='K', magnitude=4) #source pour les instruments scientifiques

ngs*tel

tel.src.print_properties()

tel.computePSF(zeroPaddingFactor = 6)
PSF_ideal = np.abs(tel.PSF)

plt.figure()
plt.imshow(np.log10(PSF_ideal))
plt.clim([-3, 4])
plt.colorbar()

#%% atmosphère

r0 = 0.5
L0 = 20
fractionalR0 = [0.5, 0.1, 0.8]
windSpeed = [5, 20, 15]
windDirection = [0, 45, 180]
altitude = [0, 2000, 10000]

atm = Atmosphere(tel, r0, L0, windSpeed, fractionalR0, windDirection, altitude) 
atm.initializeAtmosphere(tel)

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

atm.display_atm_layers()

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

#%% Décompostion de la phase

phase = atm.OPD

nmodes = 100 #nombre de modes de la décompostion
Z = Zernike(tel, nmodes)
Z.computeZernike(tel)

M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:])) @ Z.modes

dm.coefs = M2C_zernike[:,:100]
tel*dm
displayMap(tel.OPD)

Z_inv = np.linalg.pinv(Z.modes)
OPD_atm = atm.OPD[np.where(tel.pupil==1)]
coef_atm = Z_inv @ OPD_atm
OPD_atm_rec =  tel.OPD = np.squeeze(Z.modesFullRes @ coef_atm)

plt.figure()
plt.plot(range(nmodes), coef_atm)
plt.title("Décomposition de la phase de l'atmosphère")
plt.show()





















