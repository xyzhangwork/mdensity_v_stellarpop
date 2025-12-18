from astropy.io import fits
import astropy.cosmology as cosmology
import math
import numpy as np
from astropy.table import Table


cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)
mask = fits.open('sample_mask_new2.fits')
masking = fits.open('masking.fits')
manga_sga = fits.open('manga_sga_z.fits')

file_path = './data/'
sga_28 = fits.open(file_path + 'sga_mass_new_28mag2.fits')
gals = mask[1].data['plateifu'][(mask[1].data['no_sga_ellp'] == 0) &
                                (mask[1].data['no_reff'] == 0) &
                                (mask[1].data['no_sga_prof'] == 0) &
                                (mask[1].data['no_maps_url'] == 0) &
                                (mask[1].data['no_sga_url'] == 0) &
                                (mask[1].data['ba0.3'] == 0) &
                                (mask[1].data['negative_mass'] == 0) & (mask[1].data['mass11.2'] == 0) & ( mask[1].data['outliers_mlcr'] == 0) &
                                (mask[1].data['outliers_visual'] == 0) &
                                (mask[1].data['cen<psf'] == 0) &
                                (mask[1].data['small_pixnum'] == 0) &
                                (mask[1].data['morph'] == 0) &
                                (mask[1].data['reobserved'] == 0) &
                                (mask[1].data['outliers_psf'] == 0)]
rmax_manga=[]
_, idx0, idx_m = np.intersect1d(gals,sga_28[1].data['plateifu'],return_indices=True)
_, idx1, idx_z = np.intersect1d(gals,manga_sga[1].data['plateifu'],return_indices=True)
r_median = [[], [], []]
bin_snr=[]
sigma_cen,sigma_mid,sigma_out=[],[],[]
sigma_cen_noise,sigma_mid_noise,sigma_out_noise=[],[],[]
sigma_re8,sigma_re8_noise=[],[]
for gal in gals:
    print(gal)
    maps_path = 'maps path'
    idx_ml = np.where(sga_28[1].data['plateifu'] == gal)[0][0]
    idx_sga = np.where(manga_sga[1].data['plateifu'] == gal)[0][0]
    idx_mask = np.where(mask[1].data['plateifu'] == gal)[0][0]

    z_sga = manga_sga[1].data['redshift'][idx_sga]

    hdap = fits.open(maps_path + 'manga-' + gal + '-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz')
    d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value
    reff = sga_28[1].data['Re_kpc'][idx_ml]
    reff_arc = sga_28[1].data['Re_arc'][idx_ml]

    r_center = reff / 2

    ba = sga_28[1].data['ba'][idx_ml]
    ag = sga_28[1].data['pa'][idx_ml]
    Xc, Yc = hdap[0].header['objra'], hdap[0].header['objdec']
    s = math.sin(ag * math.pi / 180)
    c = math.cos(ag * math.pi / 180)
    xr = hdap[1].header['crpix1']
    yr = hdap[1].header['crpix2']
    xu = hdap[1].header['PC1_1']
    yu = hdap[1].header['PC2_2']
    Yr = Yc + hdap['SPX_SKYCOO'].data[1, int(xr), int(yr)] / 3600
    Xr = Xc + hdap['SPX_SKYCOO'].data[0, int(xr), int(yr)] / 3600
    xc0 = xr + (Xc - Xr) / xu
    yc0 = yr + (Yc - Yr) / yu
    D = hdap[1].header['NAXIS1']
    pos0 = np.full((D, D), np.arange(0, D))
    pos = np.zeros((D, D, 2))
    pos[:, :, 1] = pos0
    pos[:, :, 0] = pos0.T
    A = np.sqrt((((pos[:, :, 1] - xc0) * c + (pos[:, :, 0] - yc0) * s) / ba) ** 2 + (
            (pos[:, :, 0] - yc0) * c - (pos[:, :, 1] - xc0) * s) ** 2) * 0.5  # * (d_a * np.pi / (3600 * 180))
    A = A * (d_a * np.pi / (3600 * 180))

    A[np.isnan(hdap['SPX_SNR'].data)] = np.nan

    mask_snr = hdap['SPX_SNR'].data > 2

    bins_in = A <= reff * 0.5
    bins_out = A >= 1 * reff
    bins_mid = (A >= reff * 0.5) & (A <= reff)
    bining = np.full_like(bins_in * 1, -1)
    bining[bins_in] = 1
    bining[bins_mid] = 2
    bining[bins_out] = 3

    bining[~(hdap['SPX_SNR'].data >= 2)] = -1
    bining[hdap['STELLAR_VEL_MASK'].data >= int(2 ** 30)] = -1

    mask_out = (bining == 3)
    mask_mid = (bining == 2)
    mask_cen = (bining == 1)
    mask_1kpc = A <= 1
    mask_re8 = A <= reff / 8

    if masking['info'].data['plateifu'].__contains__(gal):
        mask_snr = mask_snr & (masking[gal].data == 0)
    rmax_manga.append(np.nanmax(A[bining > -1]))
    reff = sga_28[1].data['Re_kpc'][idx_ml]
    sigma = np.sqrt(hdap['STELLAR_SIGMA'].data ** 2 - hdap['STELLAR_SIGMACORR'].data[0, :, :] ** 2)
    sigma_noise = 1 / np.sqrt(hdap['STELLAR_SIGMA_IVAR'].data)
    mask_tmp = (~(sigma / sigma_noise > 1)) | (np.log2(hdap['STELLAR_SIGMA_MASK'].data) >= 30) | (
            np.log2(hdap['STELLAR_SIGMA_MASK'].data) == 8)
    if np.nansum((~mask_tmp) * 1) <= 1:
        print('mask shit', gal)
        mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
    sigma[mask_tmp] = np.nan
    sigma_noise[mask_tmp] = np.nan
    sigma_rmax = np.nanmax(A[~np.isnan(sigma)]) / reff
    sigma_re8.append(np.nanmean(sigma[mask_re8 == 1]))
    sigma_re8_noise.append(np.sqrt(np.nansum(sigma_noise[mask_re8 == 1] ** 2)) / np.nansum(
        1 * (~np.isnan(sigma_noise[mask_re8 == 1]))))

    sigma_cen.append(np.nanmean(sigma[mask_cen == 1]))
    sigma_cen_noise.append(np.sqrt(np.nansum(sigma_noise[mask_cen == 1] ** 2)) / np.nansum(
        1 * (~np.isnan(sigma_noise[mask_cen == 1]))))

    sigma_mid.append(np.nanmean(sigma[mask_mid == 1]))
    sigma_mid_noise.append(np.sqrt(np.nansum(sigma_noise[mask_mid == 1] ** 2)) / np.nansum(
        1 * (~np.isnan(sigma_noise[mask_mid == 1]))))

    sigma_out.append(np.nanmean(sigma[mask_out == 1]))
    sigma_out_noise.append(np.sqrt(np.nansum(sigma_noise[mask_out == 1] ** 2)) / np.nansum(
        1 * (~np.isnan(sigma_noise[mask_out == 1]))))
    r_median[0].append(np.nanmedian(A[bining == 1]))
    r_median[1].append(np.nanmedian(A[bining == 2]))
    r_median[2].append(np.nanmedian(A[bining == 3]))
    bin_snr.append(np.nanmin(hdap['BIN_SNR'].data[hdap['BIN_SNR'].data>0]))

t = Table([gals,sga_28[1].data['mass_sga_rc15'][idx_m],sga_28[1].data['lg_M/L'][idx_m],sga_28[1].data['Re_arc'][idx_m],
           sga_28[1].data['Re_kpc'][idx_m],sga_28[1].data['gr_1psf'][idx_m],sga_28[1].data['Min_10kpc'][idx_m],
           sga_28[1].data['Min_20kpc'][idx_m],sga_28[1].data['Mout_20kpc'][idx_m],sga_28[1].data['Mout_30kpc'][idx_m],
           sga_28[1].data['Min_re'][idx_m],sga_28[1].data['Mout_2re'][idx_m],
           np.log10(np.array(sigma_cen)),np.array(sigma_cen_noise)/(np.array(sigma_cen)*np.log(10)),
           np.log10(np.array(sigma_mid)),np.array(sigma_mid_noise)/(np.array(sigma_mid)*np.log(10)),
           np.log10(np.array(sigma_out)),np.array(sigma_out_noise)/(np.array(sigma_out)*np.log(10)),
           np.log10(np.array(sigma_re8)),np.array(sigma_re8_noise)/(np.array(sigma_re8)*np.log(10)),
           manga_sga[1].data['objra'][idx_z],manga_sga[1].data['objdec'][idx_z],manga_sga[1].data['redshift'][idx_z],
           manga_sga[1].data['sgaid'][idx_z],r_median[0], r_median[1], r_median[2],bin_snr],
          names=['plateifu','mass_sga_rc15','lg_M/L','Re_arc','Re_kpc','g-r','Min_10kpc',
                 'Min_20kpc','Mout_20kpc','Mout_30kpc','Min_re','Mout_2re','sigma_cen', 'sigma_cen_noise','sigma_mid', 'sigma_mid_noise',
                 'sigma_out', 'sigma_out_noise','sigma_re8', 'sigma_re8_noise','RA','DEC','z','SGAid',
                 'rin_median(kpc)', 'rmid_median(kpc)', 'rout_median(kpc)','bin_snr'])

t.write('parameters_reff2_2.fits', overwrite=True)


