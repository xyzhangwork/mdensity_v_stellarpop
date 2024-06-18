from astropy.io import fits
import astropy.cosmology as cosmology
from calc_kcor import calc_kcor
import os
import math
from scipy import interpolate
import numpy as np
from plot_cogl import get_cog
from astropy.table import Table

cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)


if __name__ == '__main__':
    file_ttype = '/scratch/shared_data/manga/MPL11_TType.fits'
    file_gzoo = '/scratch/shared_data/manga/MPL11_GalaxyZoo.fits'
    masking = fits.open('masking.fits')
    ttype_data = fits.getdata(file_ttype, 1)
    gzoo_data = fits.getdata(file_gzoo, 1)
    r_band = {'g': 3.214, 'r': 2.165, 'i': 1.592, 'z': 1.211}
    # https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients

    solar_l = {'u': 6.39, 'g': 5.11, 'r': 4.65, 'i': 4.53,
               'z': 4.5}
    # SDSS filters, from http://mips.as.arizona.edu/~cnaw/sun.html

    filter_cor = {'g': 0.0282, 'r': 0.0381,
                  'z': 0.0092}
    # filter tranformation https://transformcalc.icrar.org/

    sga = fits.open('/home/xiaoya/sample_select/SGA-2020.fits')
    manga_sga = fits.open('/home/xiaoya/sample_select/manga_sga_z.fits')
    sga_mass = fits.getdata('sga_mass_new.fits', 1)

    file = open('gal_mask.txt')
    lines = file.readlines()
    gal_mask = np.array([lines[i][:lines[i].find('_')] for i in range(len(lines))])
    mask_hdu = fits.open('/home/xiaoya/sample_select/sample_mask.fits')
    rmax_manga = np.zeros_like(mask_hdu[1].data['cen<psf'])

    sga_new_mass = sga_mass['mass_sga_rc15']
    sga_new_ifu = sga_mass['plateifu']
    _, _, idx_total = np.intersect1d(sga_new_ifu,
                                     manga_sga[1].data['plateifu'], return_indices=True)

    parameters = {'rmanga': [],
                  'ifu': [],
                  'ttype': [],
                  'gzoo': [],
                  'lumout(max-20kpc)': [],
                  'lum10kpc': [],
                  'massout(max-20kpc)': [],
                  'mass10kpc': [],
                  'snr_cen': [],
                  'snr_mid': [],
                  'snr_out': [],
                  'sigma_cen': [],
                  'sigma_cen_noise': [],
                  'sigma_mid': [],
                  'sigma_mid_noise': [],
                  'sigma_out': [],
                  'sigma_out_noise': [],
                  'sigma_rmax': [],}
    mask_names = ['no_maps',
                  'zero_maps_size',
                  'no_maps_url',
                  'no_sga_url',
                  'no_sga_ellp',
                  'no_reff',
                  'no_sga_prof',
                  'cen<psf',
                  'small_pixnum',
                  'mass11.2',
                  'outliers_mlcr',
                  'ba0.3',
                  'negative_mass',
                  'outliers_psf',
                  'outliers_visual',
                  'reobserved']
    mask = mask_hdu[1].data[mask_names[0]] != 2
    for i in range(len(mask_names)):
        mask = mask & (mask_hdu[1].data[mask_names[i]] == 0)

    factor_decalz = {'g': [0.08804478, -0.01202266],
                     'r': [0.097115220, -0.005264246],
                     'z': [0.0504410920, -0.0002501634]}
    factor_bass = {'g': [0.011121011, -0.009446287],
                   'r': [0.080892829, -0.003455136],
                   'z': [0.073761688, 0.004807630]}

    for gal in manga_sga[1].data[idx_total]:
        ifu = gal['plateifu']
        dec = gal['objdec']
        sid = gal['sgaid']
        idx_msk = np.where(mask_hdu[1].data['plateifu'] == ifu)[0][0]
        idx_ml = np.where(sga_mass['plateifu'] == ifu)[0][0]
        ml = sga_mass['lg_M/L'][idx_ml]

        idx_sga = np.where(sga[1].data['sga_id'] == sid)[0][0]
        gn = sga[1].data['group_name'][idx_sga]
        z_sga = sga[1].data['z_leda'][idx_sga]
        sid = str(sid)

        maps_path = 'manga maps file path'
        file_path='sga ellipse file path'
        hdu = fits.open(file_path + gn + '-largegalaxy-' + sid + '-ellipse.fits')
        hdap = fits.open(maps_path + 'manga-' + ifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz')

        d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value
        d_l = cosm.luminosity_distance(z_sga).to('pc').value
        rpsf = 2.5 * (d_a * np.pi / (3600 * 180))
        reff = sga_mass['Re_kpc'][idx_ml]
        r_center = reff / 2

        ebv = hdap[0].header['EBVGAL']
        ba = sga_mass['ba'][idx_ml]
        ag = sga_mass['pa'][idx_ml]
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
                (pos[:, :, 0] - yc0) * c - (pos[:, :, 1] - xc0) * s) ** 2) * 0.5  * (d_a * np.pi / (3600 * 180))

        idx_cen = np.where(hdap['SPX_ELLCOO'].data[0] == np.nanmin(hdap['SPX_ELLCOO'].data[0]))
        yc, xc = idx_cen[0][0], idx_cen[1][0]

        snr_mask = {'g': hdu[1].data['G_INTENS'][0] / hdu[1].data['G_INTENS_ERR'][0] > 3,
                    'r': hdu[1].data['R_INTENS'][0] / hdu[1].data['R_INTENS_ERR'][0] > 3,
                    'z': hdu[1].data['Z_INTENS'][0] / hdu[1].data['Z_INTENS_ERR'][0] > 3}

        sga_grz_original = {'g': get_cog(hdu[1].data['G_SMA'][0], hdu[1].data['G_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['G_EPS'][0]) * 0.263 ** 2,
                            'r': get_cog(hdu[1].data['R_SMA'][0], hdu[1].data['R_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['R_EPS'][0]) * 0.263 ** 2,
                            'z': get_cog(hdu[1].data['Z_SMA'][0], hdu[1].data['Z_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['Z_EPS'][0]) * 0.263 ** 2}
        lum10_0 = {}
        lumout20_0 = {}
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

        if masking['info'].data['plateifu'].__contains__(ifu):
            mask_snr = mask_snr & (masking[ifu].data == 0)
        rmax_manga[idx_msk] = np.nanmax(A[bining > -1])
        reff = sga_mass['Re_kpc'][idx_ml]

        for band in ['g', 'r', 'z']:
            dx = hdu[1].data[band + '_SMA'][0] * 0.263 * (d_a * np.pi / (3600 * 180))
            dx[dx < 1] = np.nan
            dx = dx[snr_mask[band]]
            dy = sga_grz_original[band][snr_mask[band]]
            dy = dy[~np.isnan(dx)]
            dx = dx[~np.isnan(dx)]
            diff = np.insert(abs(np.diff(dy)), 0, 0)
            idx_diff0 = np.where((diff > np.nanstd(dy[(dx > reff) & (dx < 1.5 * reff)])) & (dx > 1.5 * reff))[0]
            if len(idx_diff0) >= 1:
                dy[idx_diff0[0]:] = np.nan
            if np.nanmax(dx) <= 30:
                lum10_0[band] = np.nan
                lumout20_0[band] = np.nan
            else:
                func = interpolate.interp1d(dx, dy)
                lum10_0[band] = func(20)

            lum10_0[band] = 22.5 - 2.5 * np.log10(lum10_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lumout20_0[band] = 22.5 - 2.5 * np.log10(lumout20_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
        if dec < 32.375:
            lumout20_1 = {
                'g': lumout20_0['g'] + factor_decalz['g'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_decalz['g'][
                    1],
                'r': lumout20_0['r'] + factor_decalz['r'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_decalz['r'][
                    1],
                'z': lumout20_0['z'] + factor_decalz['z'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_decalz['z'][
                    1]}
            lum10_1 = {
                'g': lum10_0['g'] + factor_decalz['g'][0] * (lum10_0['g'] - lum10_0['r']) + factor_decalz['g'][1],
                'r': lum10_0['r'] + factor_decalz['r'][0] * (lum10_0['r'] - lum10_0['z']) + factor_decalz['r'][1],
                'z': lum10_0['z'] + factor_decalz['z'][0] * (lum10_0['r'] - lum10_0['z']) + factor_decalz['z'][1]}
        else:
            lumout20_1 = {
                'g': lumout20_0['g'] + factor_bass['g'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_bass['g'][1],
                'r': lumout20_0['r'] + factor_bass['r'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_bass['r'][1],
                'z': lumout20_0['z'] + factor_bass['z'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_bass['z'][1]}
            lum10_1 = {
                'g': lum10_0['g'] + factor_bass['g'][0] * (lum10_0['g'] - lum10_0['r']) + factor_bass['g'][1],
                'r': lum10_0['r'] + factor_bass['r'][0] * (lum10_0['g'] - lum10_0['r']) + factor_bass['r'][1],
                'z': lum10_0['z'] + factor_bass['z'][0] * (lum10_0['r'] - lum10_0['z']) + factor_bass['z'][1]}
        lumout20 = {
            'g': lumout20_1['g'] - calc_kcor('g', z_sga, 'gr', lumout20_1['g'] - lumout20_1['r']),
            'r': lumout20_1['r'] - calc_kcor('r', z_sga, 'gr', lumout20_1['g'] - lumout20_1['r']),
            'z': lumout20_1['r'] - calc_kcor('z', z_sga, 'gz', lumout20_1['g'] - lumout20_1['z'])
        }

        lum10 = {
            'g': lum10_1['g'] - calc_kcor('g', z_sga, 'gr', lum10_1['g'] - lum10_1['r']),
            'r': lum10_1['r'] - calc_kcor('r', z_sga, 'gr', lum10_1['g'] - lum10_1['r']),
            'z': lum10_1['r'] - calc_kcor('z', z_sga, 'gz', lum10_1['g'] - lum10_1['z'])
        }

        # k-correction: http://kcor.sai.msu.ru/
        for band in ['g', 'r', 'z']:
            lum10[band] = -0.4 * (lum10[band] - solar_l[band])
            lumout20[band] = -0.4 * (lumout20[band] - solar_l[band])

        sigma = np.sqrt(hdap['STELLAR_SIGMA'].data ** 2 - hdap['STELLAR_SIGMACORR'].data[0, :, :] ** 2)
        sigma_noise = 1 / np.sqrt(hdap['STELLAR_SIGMA_IVAR'].data)
        mask_tmp = (~(sigma / sigma_noise > 1)) | (np.log2(hdap['STELLAR_SIGMA_MASK'].data) >= 30) | (
                np.log2(hdap['STELLAR_SIGMA_MASK'].data) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        sigma[mask_tmp] = np.nan
        sigma_noise[mask_tmp] = np.nan
        sigma_rmax = np.nanmax(A[~np.isnan(sigma)]) / reff

        parameters['ifu'].append(ifu)
        idx_ttype = np.where(ttype_data['plateifu'] == ifu)[0]
        idx_gzoo = np.where(gzoo_data['plateifu'] == ifu)[0]
        if len(idx_ttype) < 1:
            parameters['ttype'].append(np.nan)
        else:
            parameters['ttype'].append(ttype_data[idx_ttype[0]]['TType'])
        if len(idx_gzoo) < 1:
            parameters['gzoo'].append(3 * [np.nan])
        else:
            parameters['gzoo'].append(gzoo_data[idx_gzoo[0]][16:])

        parameters['sigma_rmax'].append(sigma_rmax)

        mask_cen = mask_cen & mask_snr
        mask_mid = mask_mid & mask_snr
        mask_out = mask_out & mask_snr
        mask_1kpc = mask_1kpc & mask_snr
        mask_re8 = mask_re8 & mask_snr

        parameters['sigma_cen'].append(np.nanmean(sigma[mask_cen == 1]))
        parameters['sigma_cen_noise'].append(np.sqrt(np.nansum(sigma_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(sigma_noise[mask_cen == 1]))))
        parameters['sigma_mid'].append(np.nanmean(sigma[mask_mid == 1]))
        parameters['sigma_mid_noise'].append(np.sqrt(np.nansum(sigma_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(sigma_noise[mask_mid == 1]))))
        parameters['sigma_out'].append(np.nanmean(sigma[mask_out == 1]))
        parameters['sigma_out_noise'].append(np.sqrt(np.nansum(sigma_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(sigma_noise[mask_out == 1]))))

        parameters['sigma_1kpc'].append(np.nanmean(sigma[mask_1kpc == 1]))
        parameters['sigma_re_8'].append(np.nanmean(sigma[mask_re8 == 1]))

        parameters['snr_cen'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_cen == 1)] ** 2)))
        parameters['snr_mid'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_mid == 1)] ** 2)))
        parameters['snr_out'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_out == 1)] ** 2)))

        parameters['lumout(max-20kpc)'].append(lumout20['r'])
        parameters['lum10kpc'].append(lum10['r'])

        parameters['massout(max-20kpc)'].append(lumout20['r'] + ml)

        parameters['mass10kpc'].append(lum10['r'] + ml)

        parameters['rmanga'].append(np.nanmax(A[(hdap['SPX_SNR'].data > 3)] / reff))
    t = Table([parameters['ifu'], parameters['ttype'], parameters['gzoo'],
               parameters['lummassout(max-20kpc)'], parameters['lum10kpc'],
               parameters['massout(max-20kpc)'], parameters['mass10kpc'],
               parameters['snr_cen'], parameters['snr_mid'], parameters['snr_out'],
               parameters['sigma_cen'], parameters['sigma_cen_noise'],
               parameters['sigma_mid'], parameters['sigma_mid_noise'],
               parameters['sigma_out'], parameters['sigma_out_noise'],
               parameters['sigma_rmax'], parameters['sigma_1kpc'], parameters['sigma_re_8']],
              names=['plateifu', 'ttype', 'gzoo',
                     'lumout(max-20kpc)', 'lum10kpc',
                     'massout(max-20kpc)', 'mass10kpc',
                     'snr_cen', 'snr_mid', 'snr_out',
                     'sigma_cen', 'sigma_cen_noise',
                     'sigma_mid', 'sigma_mid_noise',
                     'sigma_out', 'sigma_out_noise', 'sigma_rmax',
                     'sigma_1kpc', 'sigma_re_8'])
    t.write('parameters_spx2_max_reff2.fits', overwrite=True)
