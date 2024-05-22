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

from codes_perso.Outils import buffer

# télescope

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

# source

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

# atmosphère

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

# création DM

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

# Pyramide

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


# matrice d'interractions Zernike
tel.isPaired = False
tel.resetOPD()
stroke=1e-9

M2C_KL = compute_KL_basis(tel, atm, dm, lim=0*1e-3)#lim pour controler le nombre de modes
M2C_zonal = np.eye(dm.nValidAct)

calib_zonal = InteractionMatrix(ngs=ngs, atm=atm, tel=tel, dm=dm, wfs=wfs,
                                M2C=M2C_zonal, stroke=stroke, nMeasurements=100, noise='off')

calib_KL = CalibrationVault(calib_zonal.D @ M2C_KL)

reconstructor = M2C_KL @ calib_KL.M

#camera 

camera = Detector()


#%% Application de la phase à mesurer
"""
aberration = 80e-9 * M2C_KL[:,4] + 20e-9 * M2C_KL[:, 6]

dm.coefs = 0
dm.coefs = aberration
tel*dm
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(tel.OPD)
plt.title("phase à mesurer")

#ajout de la phase de diversité
ph_div = M2C_KL[:,2] #aberration d'astigmatisme
GPD = 0*50e-9 #intensité de la phase de diversité

dm.coefs = dm.coefs + GPD * ph_div
tel*dm
plt.subplot(1, 2, 2)
plt.imshow(tel.OPD)
plt.title("Phase à mesurer + diversité")

#%% matrice d'interraction

nModes = 20 #nombre de modes considérés
amp = 10e-9

tel.computePSF(2)
[m, n] = np.shape(tel.PSF)
IM = np.zeros([m*n, nModes])

for i in range(nModes):
    Zi = M2C_KL[:, i] * amp
    
    #push
    dm.coefs = 0
    dm.coefs = GPD * ph_div + Zi
    tel*dm
    tel.computePSF(2)
    Spush = tel.PSF
    
    #pull
    dm.coefs = 0
    dm.coefs = GPD * ph_div - Zi
    tel*dm
    tel.computePSF(2)
    Spull = tel.PSF
    
    #matrice du mode
    IMzi = (Spush - Spull) / (2 * amp)
    IM[:, i] = np.ravel(IMzi)

plt.figure()
plt.imshow(IM[:, 3].reshape(tel.PSF.shape))
plt.title("Matrice d'interraction")

reconstructor = np.linalg.pinv(IM)

#%% reconstruction par LIFT

dm.coefs =  GPD * ph_div

tel*dm
tel.computePSF(2)
PSF0 = tel.PSF.ravel()

dm.coefs = aberration + GPD * ph_div

tel*dm
tel.computePSF(2)

PSF1 = tel.PSF.ravel()
estimation = np.matmul(reconstructor, PSF1)
plt.plot(estimation)

#%% matrice d'interraction

nModes = 20 #nombre de modes considérés
amp = 10e-9

estimation[2] = 0

tel.computePSF(2)
[m, n] = np.shape(tel.PSF)
IM = np.zeros([m*n, nModes])


for i in range(nModes):
    Zi = M2C_KL[:, i] * amp 
    
    #push
    dm.coefs = 0
    dm.coefs = GPD * ph_div + Zi + np.matmul(M2C_KL[:,:nModes],estimation)
    tel*dm
    tel.computePSF(2)
    Spush = tel.PSF
    
    #pull
    dm.coefs = 0
    dm.coefs = GPD * ph_div - Zi + np.matmul(M2C_KL[:,:nModes],estimation)
    tel*dm
    tel.computePSF(2)
    Spull = tel.PSF
    
    #matrice du mode
    IMzi = (Spush - Spull) / (2 * amp)
    IM[:, i] = np.ravel(IMzi)

plt.figure()
plt.imshow(IM[:, 3].reshape(tel.PSF.shape))
plt.title("Matrice d'interraction")

reconstructor2 = np.linalg.pinv(IM)

dm.coefs = GPD * ph_div + np.matmul(M2C_KL[:,:nModes],estimation)
tel*dm
tel.computePSF(2)
PSF0_2 = tel.PSF.ravel()


dm.coefs = aberration + GPD * ph_div

tel*dm
tel.computePSF(2)
PSF2 = tel.PSF.ravel()

estimation2 = np.matmul(reconstructor2, PSF2 - PSF0_2) + estimation
plt.figure()
plt.plot(estimation2)
"""

#%% LIFT en fonction

def LIFT(nIterrations, nModes, Telescope, DeformableMiror, phaseDiversity, phaseAberated, RMSamplitude, M2C_KL, zeroPaddingFactor=2):
    
    threshold = 1e-10  #threshold for convergence
    buffer_size = 5
    
    global nb_iter ##########
    M2C = M2C_KL[:, :nModes]
    Telescope.resetOPD()
    DeformableMiror.coefs = 0
    Telescope*DeformableMiror
    
    Telescope.computePSF(zeroPaddingFactor)
    
    camera.psf_sampling = zeroPaddingFactor
    ngs*tel*camera
    
    #camera.integrate(Telescope.PSF) ### apply detector noise
    #Telescope.PSF = camera.frame.copy() ### st it back to tel.PSF
    
    if affichage :
        plt.figure()
        plt.imshow(Telescope.PSF)
    
    [m, n] = np.shape(Telescope.PSF)
    IM = np.zeros([m*n, nModes]) #matrice d'interraction vide
    
    PSF0 = np.zeros([m, n])
    #buf_PSF0 = buffer(buffer_size)
    #buf_PSF1 = buffer(buffer_size)
    
    estimation = np.zeros(nModes)
    buf_estim = buffer(buffer_size, estimation)
    #estim_buffer = [estimation]*buffer_size
    PSF0 = []
    PSF1 = []
    ecarts = []
    crit = buffer(buffer_size)
    found = False                  #stop loop when convergence
    k = 0                          #itteration counter
    
    while not found :
        """print(f"progrès = {k}/{nIterrations}")"""
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
            
            #matrice du mode
            IMzi = (Spush - Spull) / (2 * RMSamplitude)
            IM[:, i] = np.ravel(IMzi)
        
        reconstructor = np.linalg.pinv(IM)
        
        DeformableMiror.coefs = phaseDiversity + M2C @ estimation
        Telescope*DeformableMiror
        Telescope.computePSF(zeroPaddingFactor)
        PSF0.append(Telescope.PSF.ravel())
        #buf_PSF0.add(tel.PSF.ravel())###
    
        DeformableMiror.coefs = phaseAberated + phaseDiversity
    
        Telescope*DeformableMiror
        Telescope.computePSF(zeroPaddingFactor)
        
        camera.integrate(Telescope.PSF) ### apply detector noise
        Telescope.PSF = camera.frame.copy() ### st it back to tel.PSF
    
        PSF1.append(tel.PSF.ravel())
        
        #estim_old = estimation.copy()
        #estimation = np.matmul(reconstructor, PSF1[-1] - PSF0[-1]) + estim_old
        #ecarts.append(ecartEstimations(estim_old, estimation))
        
        
        estimation = np.matmul(reconstructor, PSF1[-1] - PSF0[-1]) + buf_estim.access(0)
        buf_estim.add(estimation)
        
        #ecarts.append(convergenceCriterion(buf_estim))
        #crit.add(ecarts[-1])
        ecarts.append(ecartEstimations(buf_estim.access(-1), buf_estim.access(0)))
        
        #[TMP]data display
        if affichage :
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(estimation, label="estimation")
            plt.plot(PA_coefs, "--o", label="appliqué")
            plt.title(f"Estimation LIFT pour {k} iterrations")
            plt.xlabel("Indice du mode de KL")
            plt.ylabel("Amplitude du mode (m)")
            plt.legend()
    
            plt.subplot(1, 2, 2)
            plt.plot(ecarts, '-+', label="{:.2e}".format(ecarts[-1]))
            #plt.plot(np.ones(k+1)*threshold, '--', label='seuil')
            plt.title("Ecarts d'une iterration à l'autre")
            plt.xlabel("Iterration")
            plt.ylabel("Ecart")
            plt.legend()
            
            plt.tight_layout(pad=0.5)
            plt.pause(0.01)
        
        
        #testing convergence :
            
        if ecarts[-1] <= threshold :
            found = True
            print(f"LIFT sucesfully converged after {k} itterations")
            
            nb_iter.append(k) ###
        """
        if convergence(crit):
            found = True
            print(f"LIFT sucesfully converged after {k} itterations")
            
            nb_iter.append(k)"""
        #updating loop counter :
        k += 1
        if k >= nIterrations :
            nb_iter.append(k) ###
            print(f"LIFT is unable to converge after {k} itterations")
            estimation = [np.nan]*nModes
            return estimation, ecarts
            #raise ValueError(f"LIFT is unable to converge after {k} itterations")
            
            
    return estimation, ecarts



def convergenceCriterion(buffer_estimations):
    data = buffer_estimations.accessAll('raw')
    crit = np.sum(np.nanstd(data, axis=0))
    return crit

def convergence(buffer_criterion):
    threshold = 0.1
    crit = np.nanstd(buffer_criterion.accessAll()) / np.nanmean(buffer_criterion.accessAll())
    if crit < threshold :
        return True
    return False

def ecartEstimations(estim1, estim2):
    if len(estim1) != len(estim2):
        raise ValueError(f"The two estimations are different sizes {len(estim1)} and {len(estim2)}")
    #cor = np.correlate(estim1, estim2)
    ecart = np.std(estim1-estim2)
    return ecart






#%% Import des données
path = r"C:\Users\cdartevelle\Documents\Données PAPYRUS (mission OHP 26-27-28 mars)\m_kl_4\00_0.npy"
data = np.load(path)
plt.figure()
plt.imshow(np.log(data))
plt.colorbar()
plt.title("Données importées")

#%% test de la fonction

mode = 3
tstart = time.time()
nb_iter = []

amplitudes_test = np.asarray([50]) * 1e-9
amplitudes_div = np.asarray([100]) * 1e-9
res_div = []
affichage = True

#gestion du bruit
camera.photonNoise = False
camera.readoutNoise = False
ngs.nPhoton = 100000


for j in range(len(amplitudes_div)):
    res = []
    print(f"Step {j}")
    for i in range(len(amplitudes_test)):

        #PD = M2C_KL[:,13] * amplitudes_div[j]
        PD = (0*M2C_KL[:,7] + M2C_KL[:,2]) * 100e-9
        
        #ampZi = amplitudes_test[i]
        #PA_coefs = np.zeros(21)
        #PA_coefs[mode] = ampZi
        
        #PA_coefs = np.asarray([0, 50, 0, 0, 10, 0, 0, 0, 0, 80, -10, 0, 0, 20, 0, -40, 0, 60, 0, 0, 0]) * 1e-9
        PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
        PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs
        
        amp = 10e-9
        zpad = 2
        niter = 100
        
        estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
        res.append(estim[mode])
    
    print(f"temps de calcul : {time.time() - tstart} s")
    res_div.append(res)


"""
plt.figure()
plt.xlabel("Amplitude à détecter (m)")
plt.ylabel("Amplitude détectée (m)")
plt.title(f"Caractérisation du mode KL n°{mode}")
for j in range(len(amplitudes_div)):
    plt.plot(amplitudes_test, res_div[j], '-+', label="{:.2e} m de diversité".format(amplitudes_div[j]))
plt.legend()"""


#%% Rampe de défocus


nb_iter = []

amplitudes_test = np.asarray([-300, -200, -150, -100, -50, 1, 10, 25, 50, 75, 100, 150, 200, 300]) * 1e-9 #nm
amplitudes_div  = np.asarray([1/2, 1/4, 1/8, 1/16])

affichage = False

res_div = []
res_ecarts_div = []


#for D in amplitudes_div:
for i in range(5, 20, 2):
    #print(f"D = {D} λ")
    res = []
    res_ecarts = []
    for A in amplitudes_test:
        print(f"A = {A} m")
        
        PD = M2C_KL[:,2] * 1/16 * ngs.wavelength
        
        PA_coefs = np.zeros(i)
        PA_coefs[4] = A
        
        PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs
        
        amp = 10e-9
        zpad = 2
        niter = 100
        
        estim, ecarts = LIFT(nIterrations=niter, nModes=i, Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
        
        res_ecarts.append(ecartEstimations(estim, PA_coefs))
        res.append(estim[4])
    res_div.append(res)
    res_ecarts_div.append(res_ecarts)

nModes = []
for i in range(5, 20, 2):
    nModes.append(i)

plt.figure()
plt.xlabel("Amplitude à détecter (m)")
plt.ylabel("Amplitude détectée (m)")
plt.title("Caractérisation du mode KL n°4 (défocus)")
for j in range(len(res_div)):
    #plt.plot(amplitudes_test, res_div[j], '-+', label="{:.2f} λ de diversité".format(amplitudes_div[j]))
    plt.plot(amplitudes_test, res_div[j], '-+', label=f"{j} modes")
plt.legend()


plt.figure()
plt.xlabel("Amplitude à détecter (m)")
plt.ylabel("Ecart vérité/estimation (m)")
plt.yscale('log')
plt.title("Caractérisation du mode KL n°4 (défocus)")
for j in range(len(res_div)):
    #plt.plot(amplitudes_test, res_ecarts_div[j], '-+', label="{:.2f} λ de diversité".format(amplitudes_div[j]))
    plt.plot(amplitudes_test, res_ecarts_div[j], '-+', label=f"{j} modes")
plt.legend()

#%% avec plusieurs phases de diversité

affichage = False
"""
([ 2.0e-09, -1.9e-08, -3.3e-08,  4.3e-08,  3.2e-08,  5.0e-09,
       -2.2e-08, -5.0e-09, -3.7e-08,  3.1e-08,  3.8e-08, -1.8e-08,
       -4.3e-08, -3.1e-08, -9.0e-09, -1.1e-08,  4.8e-08, -1.2e-08,
        9.0e-09,  4.1e-08,  3.4e-08,  1.5e-08, -2.9e-08,  4.0e-08,
        4.3e-08, -1.7e-08, -3.9e-08, -2.5e-08,  3.0e-09,  9.0e-09])""" #PA_coefs impossibles

PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs

amp = 10e-9
zpad = 2
niter = 100

PD1 = M2C_KL[:,2] * 50e-9
PD2 = M2C_KL[:,3] * 50e-9
PD3 = M2C_KL[:,4] * 50e-9

estim1, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD1, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
estim2, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD2, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
estim3, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD3, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(estim1, label="estim, div 1")
plt.plot(estim2, label="estim, div 2")
plt.plot(estim3, label="estim, div 3")

plt.plot(PA_coefs, "--o", label="appliqué")
plt.title("Estimations LIFT")
plt.xlabel("Indice du mode de KL")
plt.ylabel("Amplitude du mode (m)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.abs(estim1-PA_coefs), label="div 1")
plt.plot(np.abs(estim2-PA_coefs), label="div 2")
plt.plot(np.abs(estim3-PA_coefs), label="div 3")
plt.title("Résidus de l'estimation")
plt.xlabel("Indice du mode de KL")
plt.ylabel("Amplitude du résidu (m)")
plt.legend()
plt.tight_layout(pad=0.5)


#%% nombre de modes de reconstruction

#resp_liste = []
#rest_liste = []
nb_iter = []
affichage = False

for k in range (10):
    resp = []
    rest = []
    PA_coefs_ref = np.random.randint(low=-50, high=50, size=30) * 1e-9
    
    amp = 10e-9
    zpad = 2
    niter = 20
    print(f"Itter {k}")
    for i in range(1, 20):
        ti = time.time()
        PD = M2C_KL[:,2] * 50e-9
        PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs
        PA_coefs = PA_coefs_ref[:i].copy()
        
        estim, ecarts = LIFT(nIterrations=niter, nModes=i, Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
        ecart = ecartEstimations(estim, PA_coefs)
        rest.append(time.time()-ti)
        resp.append(ecart)
    
    resp_liste.append(np.asarray(resp))
    rest_liste.append(np.asarray(rest))


moy = np.nanmean(resp_liste, axis=0)
moyt = np.nanmean(rest_liste, axis=0)
sig = np.nanstd(resp_liste, axis=0)
sigt = np.nanstd(rest_liste, axis=0)


plt.figure()
plt.plot(moy, '-+')
plt.plot(moy+sig, '--', color='silver')
plt.plot(moy-sig, '--', color='silver')
plt.fill_between(range(19), moy+sig, moy-sig, color='silver', alpha=0.5)
plt.title("Ecarts estimations/réalité")
plt.xlabel("Nombre de modes")
plt.ylabel("Amplitude de l'écart (m)")


plt.figure()
plt.plot(moyt, '-+')
plt.plot(moyt+sigt, '--', color='silver')
plt.plot(moyt-sigt, '--', color='silver')
plt.fill_between(range(19), moyt+sigt, moyt-sigt, color='silver', alpha=0.5)
plt.title("Ecarts estimations/réalité")
plt.title("Temps de calcul")
plt.ylabel("Temps de calcul (s)")


#%% Echantillonnage de la PSF

#resp_liste = []
#rest_liste = []
nb_iter = []
affichage = False

zpadl = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]

for k in range (10):
    resp = []
    rest = []
    PA_coefs = np.random.randint(low=-50, high=50, size=15) * 1e-9
    
    amp = 10e-9
    niter = 20
    print(f"Itter {k}")
    for i in range(len(zpadl)):
        ti = time.time()
        PD = M2C_KL[:,2] * 50e-9
        PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs
        
        zpad = zpadl[i]
        print(f"zpad = {zpad}")
        
        estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
        ecart = ecartEstimations(estim, PA_coefs)
        rest.append(time.time()-ti)
        resp.append(ecart)
    
    resp_liste.append(np.asarray(resp))
    rest_liste.append(np.asarray(rest))


moy = np.nanmean(resp_liste, axis=0)
moyt = np.nanmean(rest_liste, axis=0)
sig = np.nanstd(resp_liste, axis=0)
sigt = np.nanstd(rest_liste, axis=0)

a = 0

plt.figure()
plt.plot(zpadl[a:], moy[a:], '-+')
plt.plot(zpadl[a:], moy[a:]+sig[a:], '--', color='silver')
plt.plot(zpadl[a:], moy[a:]-sig[a:], '--', color='silver')
plt.fill_between(zpadl[a:], moy[a:]+sig[a:], moy[a:]-sig[a:], color='silver', alpha=0.5)
plt.title("Ecarts estimations/réalité")
plt.xlabel("Facteur de zéro-padding")
plt.ylabel("Amplitude de l'écart (m)")


plt.figure()
plt.plot(zpadl, moyt, '-+')
plt.plot(zpadl, moyt+sigt, '--', color='silver')
plt.plot(zpadl, moyt-sigt, '--', color='silver')
plt.fill_between(zpadl, moyt+sigt, moyt-sigt, color='silver', alpha=0.5)
plt.title("Temps de calcul")
plt.ylabel("Temps de calcul (s)")
plt.xlabel("Facteur de zéro-padding")


plt.figure()
for i in range(len(resp_liste)):
    plt.plot(zpadl, resp_liste[i], '-+')


#%% Phase de diversité utilisée


affichage = False

PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs

res = []
for i in range(20):

    PD = M2C_KL[:,i] * 50e-9
    
    #ampZi = amplitudes_test[i]
    #PA_coefs = np.zeros(21)
    #PA_coefs[mode] = ampZi
    
    #PA_coefs = np.asarray([0, 50, 0, 0, 10, 0, 0, 0, 0, 80, -10, 0, 0, 20, 0, -40, 0, 0, 0, 0, 0]) * 1e-9
    
    
    amp = 10e-9
    zpad = 2
    niter = 20
    
    estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
    res.append(ecartEstimations(estim, PA_coefs))


plt.figure()
plt.xlabel("Ecart estimation/réalité (m)")
plt.ylabel("Indice du mode KL")
plt.title("Influence du mode de diversité")
plt.plot(res, '-+')

#%% Amplitude phase de diversité

affichage = False

PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs

res = []
for i in range(20):

    PD = M2C_KL[:,i] * 50e-9
    
    #ampZi = amplitudes_test[i]
    #PA_coefs = np.zeros(21)
    #PA_coefs[mode] = ampZi
    
    #PA_coefs = np.asarray([0, 50, 0, 0, 10, 0, 0, 0, 0, 80, -10, 0, 0, 20, 0, -40, 0, 0, 0, 0, 0]) * 1e-9
    
    
    amp = 10e-9
    zpad = 2
    niter = 20
    
    estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
    res.append(ecartEstimations(estim, PA_coefs))


plt.figure()
plt.xlabel("Ecart estimation/réalité (m)")
plt.ylabel("Indice du mode KL")
plt.title("Influence du mode de diversité")
plt.plot(res, '-+')


#%% Amplitude de la phase de diversité
mode = 2

affichage = False

PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs

amplitudes_div = np.asarray([10, 25, 40, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700]) * 1e-9

nb_iter = []

res = []
for i in range(len(amplitudes_div)):

    PD = M2C_KL[:, mode] * amplitudes_div[i]
    
    amp = 10e-9
    zpad = 2
    niter = 20
    
    estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
    res.append(ecartEstimations(estim, PA_coefs))


plt.figure()
plt.subplot(2, 1, 1)
plt.ylabel("Ecart estimation/réalité (m)")
plt.xlabel("Ampltude de la diversité (m)")
plt.title("Influence de l'amplitude de la diversité")
plt.plot(amplitudes_div, res, '-+')

plt.subplot(2, 1, 2)
plt.ylabel("Nombre d'ittérations")
plt.xlabel("Ampltude de la diversité (m)")
plt.plot(amplitudes_div, nb_iter, '-+')

#%% Amplitude RMS
mode = 2

affichage = False

PA_coefs = np.random.randint(low=-50, high=50, size=20) * 1e-9
PA = M2C_KL[:, :len(PA_coefs)] @ PA_coefs

amplitudes_RMS = np.asarray([1, 5, 10, 15, 20, 30, 50, 75, 100]) * 1e-9

nb_iter = []

res = []
for i in range(len(amplitudes_RMS)):

    PD = M2C_KL[:, mode] * 100e-9
    
    amp = amplitudes_RMS[i]
    zpad = 2
    niter = 20
    
    estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
    res.append(ecartEstimations(estim, PA_coefs))


plt.figure()
plt.subplot(2, 1, 1)
plt.ylabel("Ecart estimation/réalité (m)")
plt.xlabel("Amplitude RMS (m)")
plt.title("Influence de l'amplitude RMS")
plt.plot(amplitudes_RMS, res, '-+')

plt.subplot(2, 1, 2)
plt.ylabel("Nombre d'ittérations")
plt.xlabel("Amplitude RMS (m)")
plt.plot(amplitudes_RMS, nb_iter, '-+')


#%% anges du dm

mode = 3
tstart = time.time()
nb_iter = []

affichage = False

#gestion du bruit
camera.photonNoise = False
camera.readoutNoise = False
ngs.nPhoton = 100000

from parameter_files.parameterFile_simple_Papyrus import initializeParameterFile
import codes_perso.LIFT

param = initializeParameterFile()
#PA_coefs = np.random.randint(low=-50, high=50, size=10) * 1e-9
PD = M2C_KL[:,2] * 100e-9
PA = M2C_KL[:, 5] * 50 * 1e-9 + PD #@ PA_coefs

dm.coefs = 0
dm.coefs = PA
tel.computePSF()
plt.imshow(tel.PSF)
plt.figure()

for i in range(20):
    
    print("angle = ", i*5)
    
    misReg = MisRegistration()
    misReg.rotationAngle = i*5
    dm = DeformableMirror(telescope=tel, nSubap=nAct-1, mechCoupling=mechCoupling, misReg=misReg, coordinates=None, pitch=pitch)
    
    #[création PSF]
    
    misReg = MisRegistration()
    misReg.rotationAngle = 30
    dm = DeformableMirror(telescope=tel, nSubap=nAct-1, mechCoupling=mechCoupling, misReg=misReg, coordinates=None, pitch=pitch)
    
    amp = 10e-9
    zpad = 2
    niter = 20
    
    estim, ecarts = LIFT(nIterrations=niter, nModes=PA_coefs.shape[0], Telescope=tel, DeformableMiror=dm, phaseDiversity=PD, phaseAberated=PA, RMSamplitude=amp, M2C_KL=M2C_KL, zeroPaddingFactor=zpad)
    plt.plot(estim, label=f"Angle dm = {5*i}°")
    
print(f"temps de calcul : {time.time() - tstart} s")

plt.xlabel("Indice du mode de KL")
plt.ylabel("Amplitude détectée (m)")
plt.title(f"Influence de l'angle du DM")
plt.legend()







