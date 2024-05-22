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
from OOPAO.Detector import Detector

from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
from OOPAO.Pyramid import Pyramid


#%% télescope

n_subaperture = 50
res = 6*n_subaperture
diam = 10

tel = Telescope(resolution=res, diameter=diam, samplingTime=1/1000, centralObstruction=0.2, display_optical_path=True)


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
nAct = 20
pitch = tel.D/nAct

dm = DeformableMirror(telescope=tel, nSubap=nAct-1, mechCoupling=mechCoupling, misReg=misReg, coordinates=None, pitch=pitch)


plt.figure()
plt.imshow(tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'+')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('Position actuateurs DM')

#%% Pyramide

modulation = 3 #[λ/D]
lightRatio = 0.1
tel.resetOPD()

wfs = Pyramid(nSubap=n_subaperture, telescope=tel, modulation=modulation,
              lightRatio=lightRatio, n_pix_separation=2)

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('Pixels du WFS')

plt.figure()
plt.imshow(wfs.signal_2D)
plt.title("Signal mesuré par le WFS")
plt.colorbar()


#%% matrice d'interractions Zernike
tel.isPaired = False
tel.resetOPD()
stroke=1e-9

M2C_KL = compute_KL_basis(tel, atm, dm, lim=0*1e-3)#lim pour controler le nombre de modes
M2C_zonal = np.eye(dm.nValidAct)

calib_zonal = InteractionMatrix(ngs=ngs, atm=atm, tel=tel, dm=dm, wfs=wfs,
                                M2C=M2C_zonal, stroke=stroke, nMeasurements=100, noise='off')

calib_KL = CalibrationVault(calib_zonal.D @ M2C_KL)

reconstructor = M2C_KL @ calib_KL.M


#%% Application de la phase à mesurer



dm.coefs = 0
dm.coefs = M2C_KL[:,:16]
tel*dm
displayMap(tel.OPD)









