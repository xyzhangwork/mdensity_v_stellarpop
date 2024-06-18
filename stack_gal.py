import numpy as np
from astropy.io import fits
import math
import astropy.constants
from mangadap.config import manga, defaults
from mangadap.datacube import MaNGADataCube
from mangadap.proc.spectralstack import SpectralStack
from unred_curve import ccm_unred
import os
from astropy.table import Table
import astropy.cosmology as cosmology


def get_redshift(plate, ifu, drpall_file=None):
    """
    Get the redshift of a galaxy from the DRPall file.

    Args:
        plate (:obj:`int`):
            Plate number
        ifu (:obj:`int`):
            IFU identifier
        drapall_file (:obj:`str`, optional):
            DRPall file. If None, attempts to use the default path to
            the file using environmental variables.

    Returns:
        :obj:`float`: The redshift to the galaxy observed by the
        provided PLATEIFU.
    """
    if drpall_file is None:
        drpall_file = manga.drpall_file()
    if not drpall_file.exists():
        raise FileNotFoundError(f'Could not find DRPall file: {drpall_file}')
    hdu = fits.open(str(drpall_file))
    indx = hdu[1].data['PLATEIFU'] == '{0}-{1}'.format(plate, ifu)
    return hdu[1].data['NSA_Z'][indx][0]


C = astropy.constants.c.to('km/s').value
sga_mass = fits.open('sga_mass_new.fits')
masking = fits.open('masking.fits')
parameters = fits.open('parameters_spx2_max_reff2.fits')

specstack = SpectralStack()

gals = parameters[1].data['plateifu']
stacked = fits.open('stacked_smooth_sigma300_mask.fits')
plateifus = stacked['info'].data['plateifu']

size = len(gals)
wave_size = 4563
flux_gals_in, flux_gals_mid, flux_gals_out = np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size))

sres_gals_in, sres_gals_mid, sres_gals_out = np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size))

ivar_gals_in, ivar_gals_mid, ivar_gals_out = np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size)), \
                                             np.ma.zeros((size, wave_size))

std_gals_in, std_gals_mid, std_gals_out = np.ma.zeros((size, wave_size)), \
                                          np.ma.zeros((size, wave_size)), \
                                          np.ma.zeros((size, wave_size))

waves_tmp = []
redshifts = []
drpver = 'v3_1_1'
dprall_path = defaults.Path('/scratch/shared_data/manga/')
drpall_file = dprall_path / f'drpall-{drpver}.fits'
snrs = []
cats = []
fail_cat = []
cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)
r_median = [[], [], []]
snr_median = [[], [], []]
reffs=[]
for i, gal in enumerate(plateifus):
    plate = gal[0:gal.find('-')]
    ifu = gal[gal.find('-') + 1:]
    maps_path = 'maps path'
    logs_path = 'logcube path'
    idx0 = np.where(sga_mass[1].data['plateifu'] == gal)[0]
    directory_path = defaults.Path(logs_path)

    z0 = get_redshift(plate, ifu, drpall_file)
    redshifts.append(z0)
    spx = fits.open(maps_path + 'manga-' + str(plate) + '-' + str(ifu) + '-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz')
    cube = MaNGADataCube.from_plateifu(plate, ifu, directory_path=directory_path)  # / 'test')
    if cube.fluxhdr['NAXIS1'] != spx[1].header['NAXIS1']:
        flux_gals_in[i] = np.full(len(flux_gals_in[i]), np.nan)
        flux_gals_mid[i] = np.full(len(flux_gals_mid[i]), np.nan)
        flux_gals_out[i] = np.full(len(flux_gals_out[i]), np.nan)
        cats.append(gal)
        fail_cat.append(gal)
        continue
    ebv = spx[0].header['EBVGAL']
    ba = sga_mass[1].data['ba'][idx0]
    ag = sga_mass[1].data['pa'][idx0]
    Xc, Yc = spx[0].header['objra'], spx[0].header['objdec']
    s = math.sin(ag * math.pi / 180)
    c = math.cos(ag * math.pi / 180)
    xr = spx[1].header['crpix1']
    yr = spx[1].header['crpix2']
    xu = spx[1].header['PC1_1']
    yu = spx[1].header['PC2_2']
    Yr = Yc + spx['SPX_SKYCOO'].data[1, int(xr), int(yr)] / 3600
    Xr = Xc + spx['SPX_SKYCOO'].data[0, int(xr), int(yr)] / 3600
    xc = xr + (Xc - Xr) / xu
    yc = yr + (Yc - Yr) / yu
    D = spx[1].header['NAXIS1']
    pos0 = np.full((D, D), np.arange(0, D))
    pos = np.zeros((D, D, 2))
    pos[:, :, 1] = pos0
    pos[:, :, 0] = pos0.T

    A = np.sqrt((((pos[:, :, 1] - xc) * c + (pos[:, :, 0] - yc) * s) / ba) ** 2 + (
            (pos[:, :, 0] - yc) * c - (pos[:, :, 1] - xc) * s) ** 2) * 0.5

    reff = sga_mass[1].data['Re_kpc'][idx0]
    d_a = cosm.angular_diameter_distance(z0).to('kpc').value
    d_u = (d_a * np.pi / (3600 * 180))
    A = A * (d_a * np.pi / (3600 * 180))
    bins_in = A <= reff * 0.5
    bins_out = A >= 1 * reff
    bins_mid = (A >= reff * 0.5) & (A <= reff)
    bining = np.full_like(bins_in * 1, -1)
    bining[bins_in] = 1
    bining[bins_mid] = 2
    bining[bins_out] = 3
    bining[~(spx['SPX_SNR'].data >= 2)] = -1
    bining[spx['STELLAR_VEL_MASK'].data >= int(2 ** 30)] = -1
    if masking['info'].data['plateifu'].__contains__(gal):
        bining[masking[gal].data != 0] = -1
    r_median[0].append(np.nanmedian(A[bining == 1]))
    r_median[1].append(np.nanmedian(A[bining == 2]))
    r_median[2].append(np.nanmedian(A[bining == 3]))
    snr_median[0].append(np.nanmedian(spx['SPX_SNR'].data[bining == 1]))
    snr_median[1].append(np.nanmedian(spx['SPX_SNR'].data[bining == 2]))
    snr_median[2].append(np.nanmedian(spx['SPX_SNR'].data[bining == 3]))
    reffs.append(reff)

    if np.nansum(1 * (bining == 2)) <= 1 or np.nansum(1 * (bining == 3)) <= 1 or np.nansum(1 * (bining == 1)) <= 1:
        flux_gals_in[i] = np.full(len(flux_gals_in[i]), np.nan)
        flux_gals_mid[i] = np.full(len(flux_gals_mid[i]), np.nan)
        flux_gals_out[i] = np.full(len(flux_gals_out[i]), np.nan)
        cats.append(gal)
        fail_cat.append(gal)
        continue
    vel = spx['STELLAR_VEL'].data
    z = vel / astropy.constants.c.to('km/s').value
    pars = {'operation': 'mean', 'register': True, 'cz': astropy.constants.c.to('km/s').value * z.flatten(),
            'covar_mode': None, 'covar_par': None}
    flux = cube.copy_to_masked_array(flag=cube.do_not_stack_flags())
    ivar = cube.copy_to_masked_array(attr='ivar', flag=cube.do_not_stack_flags())
    sres = cube.copy_to_array(attr='sres')
    covar = SpectralStack.build_covariance_data(cube, pars['covar_mode'], pars['covar_par'])

    r_flux = specstack.stack(cube.wave, flux, operation=pars['operation'], binid=bining.flatten(), ivar=ivar,
                             sres=sres, cz=pars['cz'],
                             log=True, covariance_mode=pars['covar_mode'], covar=covar, keep_range=True)
    flux_ = ccm_unred(r_flux[0], ebv, flux=r_flux[1].data)

    flux_in, flux_mid, flux_out = np.copy(flux_[0]), np.copy(flux_[1]), np.copy(flux_[2])
    sres_in, sres_mid, sres_out = np.copy(r_flux[5][0]), np.copy(r_flux[5][1]), np.copy(r_flux[5][2])
    ivar_in, ivar_mid, ivar_out = np.copy(r_flux[4][0].data), np.copy(r_flux[4][1].data), np.copy(r_flux[4][2].data)
    std_in, std_mid, std_out = np.copy(r_flux[2].data[0]), np.copy(r_flux[2].data[1]), np.copy(r_flux[2].data[2])

    flux_gals_in[i] = flux_in
    flux_gals_mid[i] = flux_mid
    flux_gals_out[i] = flux_out

    sres_gals_in[i] = sres_in
    sres_gals_mid[i] = sres_mid
    sres_gals_out[i] = sres_out

    ivar_gals_in[i] = ivar_in
    ivar_gals_mid[i] = ivar_mid
    ivar_gals_out[i] = ivar_out

    std_gals_in[i] = std_in
    std_gals_mid[i] = std_mid
    std_gals_out[i] = std_out

    waves_tmp.append(r_flux[0])
    cats.append(gal)
newwave_in, _flux_in, _ivar_in, _sres_in = specstack.register(waves_tmp[0],
                                                              astropy.constants.c.to('km/s').value * np.array(
                                                                  redshifts),
                                                              flux_gals_in, ivar=ivar_gals_in, sres=sres_gals_in,
                                                              keep_range=True, log=True)
newwave_mid, _flux_mid, _ivar_mid, _sres_mid = specstack.register(waves_tmp[0],
                                                                  astropy.constants.c.to('km/s').value * np.array(
                                                                      redshifts),
                                                                  flux_gals_mid, ivar=ivar_gals_mid, sres=sres_gals_mid,
                                                                  keep_range=True, log=True)
newwave_out, _flux_out, _ivar_out, _sres_out = specstack.register(waves_tmp[0],
                                                                  astropy.constants.c.to('km/s').value * np.array(
                                                                      redshifts),
                                                                  flux_gals_out, ivar=ivar_gals_out, sres=sres_gals_mid,
                                                                  keep_range=True, log=True)

_, _, _std_in, _ = specstack.register(waves_tmp[0],
                                      astropy.constants.c.to('km/s').value * np.array(redshifts),
                                      flux_gals_in, ivar=1 / np.square(std_gals_in), keep_range=True, log=True)
_, _, _std_mid, _ = specstack.register(waves_tmp[0],
                                       astropy.constants.c.to('km/s').value * np.array(redshifts),
                                       flux_gals_mid, ivar=1 / np.square(std_gals_mid), keep_range=True, log=True)
_, _, _std_out, _ = specstack.register(waves_tmp[0],
                                       astropy.constants.c.to('km/s').value * np.array(redshifts),
                                       flux_gals_out, ivar=1 / np.square(std_gals_out), keep_range=True, log=True)

fail_cat = np.array(fail_cat)
primary_hdu = fits.PrimaryHDU(np.zeros(2))
print(np.nansum(newwave_mid - newwave_in), 'delta wave in mid')
print(np.nansum(newwave_out - newwave_in), 'delta wave in out')

w_in = fits.ImageHDU(newwave_in, name='wave')
in_tmp, mid_tmp, out_tmp = np.zeros((size - len(fail_cat), wave_size)), \
                           np.zeros((size - len(fail_cat), wave_size)), \
                           np.zeros((size - len(fail_cat), wave_size))

in_ivar_tmp, mid_ivar_tmp, out_ivar_tmp = np.zeros((size - len(fail_cat), wave_size)), \
                                          np.zeros((size - len(fail_cat), wave_size)), \
                                          np.zeros((size - len(fail_cat), wave_size))

in_sres_tmp, mid_sres_tmp, out_sres_tmp = np.zeros((size - len(fail_cat), wave_size)), \
                                          np.zeros((size - len(fail_cat), wave_size)), \
                                          np.zeros((size - len(fail_cat), wave_size))

in_std_tmp, mid_std_tmp, out_std_tmp = np.zeros((size - len(fail_cat), wave_size)), \
                                       np.zeros((size - len(fail_cat), wave_size)), \
                                       np.zeros((size - len(fail_cat), wave_size))

cats_tmp = []
j = 0
for i in range(len(cats)):
    if fail_cat.__contains__(cats[i]):
        continue
    in_tmp[j] = _flux_in.data[i]
    mid_tmp[j] = _flux_mid.data[i]
    out_tmp[j] = _flux_out.data[i]

    in_sres_tmp[j] = _sres_in[i]
    mid_sres_tmp[j] = _sres_mid[i]
    out_sres_tmp[j] = _sres_out[i]

    in_ivar_tmp[j] = _ivar_in.data[i]
    mid_ivar_tmp[j] = _ivar_mid.data[i]
    out_ivar_tmp[j] = _ivar_out.data[i]

    in_std_tmp[j] = 1 / np.sqrt(_std_in.data[i])
    mid_std_tmp[j] = 1 / np.sqrt(_std_mid.data[i])
    out_std_tmp[j] = 1 / np.sqrt(_std_out.data[i])

    cats_tmp.append(cats[i])
    j += 1
plateifus = fits.Column(name='plateifu', array=np.array(cats_tmp), format='32A')
cols = fits.ColDefs([plateifus])
pltifu = fits.BinTableHDU.from_columns(cols, name='info')

f_in = fits.ImageHDU(in_tmp, name='flux_in')
f_mid = fits.ImageHDU(mid_tmp, name='flux_mid')
f_out = fits.ImageHDU(out_tmp, name='flux_out')

r_in = fits.ImageHDU(in_sres_tmp, name='sres_in')
r_mid = fits.ImageHDU(mid_sres_tmp, name='sres_mid')
r_out = fits.ImageHDU(out_sres_tmp, name='sres_out')

i_in = fits.ImageHDU(in_ivar_tmp, name='ivar_in')
i_mid = fits.ImageHDU(mid_ivar_tmp, name='ivar_mid')
i_out = fits.ImageHDU(out_ivar_tmp, name='ivar_out')

s_in = fits.ImageHDU(in_std_tmp, name='std_in')
s_mid = fits.ImageHDU(mid_std_tmp, name='std_mid')
s_out = fits.ImageHDU(out_std_tmp, name='std_out')

hdul = fits.HDUList([primary_hdu, pltifu, w_in,
                     f_in, f_mid, f_out,
                     r_in, r_mid, r_out,
                     i_in, i_mid, i_out,
                     s_in, s_mid, s_out])
hdul.writeto('stacked_spec.fits', overwrite=True)
