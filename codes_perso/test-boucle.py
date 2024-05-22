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

#%%

tel.isPaired = False
tel.resetOPD()

wfs = ShackHartmann(nSubap=n_subaperture, telescope=tel, lightRatio=0.5,
                    binning_factor=1, is_geometric=False, shannon_sampling=True)

#%% matrice d'interractions Zernike

#commenter le mode à ne pas utiliser
#mode = 'Zernike'
mode = 'KL'

if mode=='Zernike':
    modes = 100 #nombre de modes de Zernike utilisés
    
    tel.isPaired = False
    tel.resetOPD()
    wfs.is_geometric = True
    
    Z = Zernike(tel, modes)
    Z.computeZernike(tel)
    
    M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes
    M2C_modal = M2C_zernike
    stroke=1e-9
    calib_zernike = InteractionMatrix(ngs=ngs, atm=atm, tel=tel, dm=dm, wfs=wfs,
                                    M2C=M2C_modal, stroke=stroke, nMeasurements=100, noise='off')
    
    reconstructor = M2C_zernike @ calib_zernike.M
    
    tel.isPaired = False
    tel.resetOPD()
    wfs.is_geometric = False

elif mode=='KL':
    wfs.is_geometric = True
    tel.isPaired = False
    tel.resetOPD()
    stroke=1e-9
    
    M2C_KL = compute_KL_basis(tel, atm, dm, lim=0*1e-3)#lim pour controler le nombre de modes
    M2C_zonal = np.eye(dm.nValidAct)
    
    calib_zonal = InteractionMatrix(ngs=ngs, atm=atm, tel=tel, dm=dm, wfs=wfs,
                                    M2C=M2C_zonal, stroke=stroke, nMeasurements=100, noise='off')
    
    calib_KL = CalibrationVault(calib_zonal.D @ M2C_KL)

    
    wfs.is_geometric = False
    
    #%%
modes = 328 #nombre de modes de Zernike utilisés

Z = Zernike(tel, modes)
Z.computeZernike(tel)

M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes
M2C_modal = M2C_zernike
#%% Boucle d'OA


calib_KL = CalibrationVault(calib_zonal.D @ M2C_KL[:,:])
calib_Zernike = CalibrationVault(calib_zonal.D @ M2C_zernike[:,:])


reconstructor = M2C_zonal @ calib_zonal.M
# reconstructor = M2C_zernike[:,:] @ calib_Zernike.M
# reconstructor = M2C_KL @ calib_KL.M

liste = []
iterations = 500
plt.figure()
tel+atm
dm.coefs = 0
tel.computePSF(4)
phase_var = []

for i in range(iterations):
    #changement des conditions atm
    atm.update()
    
    #propagation de la lumière
    ngs*tel*dm*wfs
    
    #actualisation du miroir
    dm_commands = np.matmul(reconstructor, wfs.signal)
    dm.coefs = dm.coefs - 0.5*dm_commands
    tel.computePSF(4)
    phase_var.append(np.std(tel.OPD[np.where(tel.pupil>0)])*1e9)
    
    #affichage des résultats
    plt.subplot(2, 3, 1)
    plt.imshow(atm.OPD*1e9)
    plt.title('Turbulence atm')
    
    plt.subplot(2, 3, 2)
    plt.imshow(tel.src.phase)
    plt.title("Phase Residuelle")
    
    plt.subplot(2, 3, 3)
    plt.plot(range(i+1), phase_var, '-+')
    plt.xlabel("ittération")
    plt.ylabel("rad")
    plt.title("RMS phase PSF")
    
    plt.subplot(2, 3, 4)
    plt.imshow(wfs.cam.frame)
    plt.colorbar()
    plt.title('Pixels du WFS')

    plt.subplot(2, 3, 5)
    plt.imshow(np.log(tel.PSF))
    plt.clim([-3, 6])
    plt.colorbar()
    plt.title(f"PSF à i = {i}/{iterations}")
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(PSF_ideal))
    plt.clim([-3, 6])
    plt.colorbar()
    plt.title("PSF sans atm")
    
    plt.tight_layout(pad=0.5)
    plt.pause(0.01)
















