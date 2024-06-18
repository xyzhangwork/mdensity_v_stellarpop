from symbol import parameters

from astropy.io import fits
import astropy.cosmology as cosmology
from calc_kcor import calc_kcor
import matplotlib.pyplot as plt
import os
import math
from scipy import interpolate
import sys
import numpy as np
from plot_cogl import get_cog
from astropy.table import Table

import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)


def cal_gas(wave_tmp, ha_tmp, hb_tmp, Rv=4.1):
    ebv_g = 0.935 * np.log((ha_tmp / hb_tmp) / 2.88)
    x = 10000.0 / wave_tmp
    if wave_tmp >= 6300 and wave_tmp <= 22000:
        klam = 2.659 * (-1.857 + 1.040 * x) + Rv
    elif wave_tmp >= 912 and wave_tmp < 6300:
        c2 = np.array([-2.156, 1.509, -0.198, 0.011])
        p2 = np.poly1d(c2[::-1])
        klam = 2.659 * p2(x) + Rv
    return (0.4 * klam * ebv_g)


def emlines_sb(z):
    # z: redshift
    arcsec_per_spaxel = 0.5
    d_l = cosm.luminosity_distance(z).to('cm').value  # cm
    d_a = cosm.angular_diameter_distance(z).value * 1e3  # kpc
    return 1e-17 * 4 * np.pi * d_l ** 2 / (arcsec_per_spaxel / 3600 / 180 * np.pi * d_a) ** 2


if __name__ == '__main__':
    file_ttype = '/scratch/shared_data/manga/MPL11_TType.fits'
    file_gzoo = '/scratch/shared_data/manga/MPL11_GalaxyZoo.fits'
    masking = fits.open('masking.fits')
    ttype_data = fits.getdata(file_ttype, 1)
    gzoo_data = fits.getdata(file_gzoo, 1)
    ns = [3, 4, 5]
    r_max = [[], []]
    reffs = []
    cen_num = 1  # int(sys.argv[1])
    cens = ['10kpc',
            'reff2']
    # r_band = {'g': 3.237, 'r': 2.176, 'i': 1.595, 'z': 1.217}
    r_band = {'g': 3.214, 'r': 2.165, 'i': 1.592, 'z': 1.211}
    # https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients

    solar_l = {'u': 6.39, 'g': 5.11, 'r': 4.65, 'i': 4.53,
               'z': 4.5}  # SDSS filters, from http://mips.as.arizona.edu/~cnaw/sun.html
    filter_cor = {'g': 0.0282, 'r': 0.0381,
                  'z': 0.0092}  # filter tranformation https://transformcalc.icrar.org/

    sga = fits.open('/home/xiaoya/sample_select/SGA-2020.fits')
    mass_bin = np.load('mass_bin.npy', allow_pickle=True)
    manga_sga = fits.open('/home/xiaoya/sample_select/manga_sga_z.fits')
    sga_mass = fits.getdata('sga_mass_new.fits', 1)
    '''
    outliers = np.array(['10837-12703', '10842-9102', '11014-12701', '11865-12705',
                         '11962-12702', '11970-6104', '12081-9101', '12489-12704',
                         '12495-12705', '12684-12701', '7968-12705', '7968-6102',
                         '7981-12705', '8077-12703', '8621-12705', '9041-1901',
                         '9086-12701', '9091-12702', '9499-6104', '9886-1901'])
    '''
    outliers = np.array(['10495-6101', '10837-12703', '10842-9102', '11865-12705',
                         '11962-12702', '11984-12704', '12484-12705', '12489-12704',
                         '12495-12705', '12684-12701', '7968-6102', '8077-12703',
                         '8131-3701', '8256-12704', '8274-12704', '8309-3702',
                         '8444-9101', '8451-12701', '8601-3702', '8621-12705',
                         '9024-12705', '9044-12701', '9086-12701', '9886-1901'])
    outliers_mass = np.array(['8091-1902', '8314-12703', '8993-12703', '9188-12704'])
    outliers_psf = np.array(['11025-1902', '11952-1901', '8462-1902', '8553-1901',
                             '8561-1902', '9035-1902', '9191-3701'])
    reobserved = np.array(['9884-1902', '8456-6104', '12511-3703', '12667-1902', '7963-6101',
                           '9036-1901', '11838-3703', '8329-3703', '8274-6104', '8451-6103',
                           '11950-1902', '9031-3704', '8274-6103', '8451-6102'])
    # outliers_visual = np.array(['10223-3701', '10493-6103', '10838-12702', '10839-6104', '10840-6103',
    #                             '11020-12705', '11020-9101', '11024-12702', '11745-12701', '11752-6101',
    #                             '11752-6103', '11757-9102', '11830-9101', '11833-9101', '11838-3703',
    #                             '11865-9101', '11866-12704', '11948-6103', '11951-6102', '11974-12705',
    #                             '11981-12705', '12518-6104', '12678-12703', '12682-9101', '12684-9102',
    #                             '8082-12705', '8095-6101', '8132-6103', '8146-12704', '8154-12702',
    #                             '8156-9101', '8157-3702', '8244-1901', '8253-12702', '8257-6104', '8327-3704',
    #                             '8601-12704',
    #                             '8555-6103', '8600-9101', '8613-9102', '8615-12704', '8622-9101',
    #                             '8952-12702', '8983-12704', '8986-12704', '9025-6102', '9032-3702',
    #                             '9039-12703', '9048-3703', '9086-6101', '9088-9102', '9092-12702',
    #                             '9094-12704', '9196-12705', '9196-6102', '9484-12702', '9887-12704',
    #                             '9887-1902', '9888-12703'])
    outliers_spec = np.array(['11980-6103', '11980-6104', '8613-12705', '8995-12704', '9034-6104'])
    file = open('gal_redo.txt')
    lines = file.readlines()
    redo = np.array([lines[i][:lines[i].find('_')] for i in range(len(lines))])

    file = open('gal_discard.txt')
    lines = file.readlines()
    discard = np.array([lines[i][:lines[i].find('_')] for i in range(len(lines))])

    remove = np.insert(redo, 0, discard)

    file = open('gal_mask.txt')
    lines = file.readlines()
    gal_mask = np.array([lines[i][:lines[i].find('_')] for i in range(len(lines))])
    _, _, idx_ = np.intersect1d(gal_mask, remove, return_indices=True)
    outliers_visual = np.delete(remove, idx_)
    # _, idx_, _ = np.intersect1d(gals, remove, return_indices=True)

    # sga = fits.open('/home/xiaoya/sample_select/SGA-2020.fits')
    p = fits.open('/home/xiaoya/sample_select/SDSS17Pipe3D_v3_1_1.fits')
    mask_hdu = fits.open('/home/xiaoya/sample_select/sample_mask.fits')
    rmax_manga = np.zeros_like(mask_hdu[1].data['cen<psf'])

    # mask_coord = manga_sga[1].data['objdec'] < 30
    # mask_mass = manga_sga[1].data['m'] > 11.2
    lum_dist = []
    # _, _, idx_total = np.intersect1d(lum_tot.T[0], manga_sga[1].data['plateifu'], return_indices=True)
    _, _, idx_outliers = np.intersect1d(outliers, sga_mass['plateifu'], return_indices=True)
    sga_new_mass = sga_mass['mass_sga_rc15']  # np.delete(sga_mass['mass_sga_rc15'], idx_outliers)
    sga_new_ifu = sga_mass['plateifu']  # np.delete(sga_mass['plateifu'], idx_outliers)
    _, _, idx_total = np.intersect1d(sga_new_ifu,  # [sga_new_mass >= 11.2],
                                     manga_sga[1].data['plateifu'], return_indices=True)

    # width = int(len(lum_tot) / n)
    # lum_bins = np.empty((n, 0)).tolist()
    # lum_lims = np.empty((n, 0)).tolist()
    parameters = {'rmanga': [],
                  'ifu': [],
                  'ttype': [],
                  'gzoo': [],
                  'lummax': [],
                  'lumout(max-30kpc)': [],
                  'lum20kpc': [],
                  'lumout(max-20kpc)': [],
                  'lum10kpc': [],
                  'lum_cen': [],
                  'lumout_re': [],
                  'lum_mid': [],
                  'massmax': [],
                  'massout(max-30kpc)': [],
                  'mass20kpc': [],
                  'massout(max-20kpc)': [],
                  'mass10kpc': [],
                  'mass_cen': [],
                  'massout_re': [],
                  'mass_mid': [],
                  'snr_cen': [],
                  'snr_mid': [],
                  'snr_out': [],
                  'all_cen': [],
                  'all_mid': [],
                  'all_out': [],
                  'taken_cen': [],
                  'taken_mid': [],
                  'taken_out': [],
                  'sigma_cen': [],
                  'sigma_cen_noise': [],
                  'sigma_mid': [],
                  'sigma_mid_noise': [],
                  'sigma_out': [],
                  'sigma_out_noise': [],
                  'sigma_rmax': [],
                  'ca4227_cen': [],
                  'ca4227_cen_noise': [],
                  'ca4227_mid': [],
                  'ca4227_mid_noise': [],
                  'ca4227_out': [],
                  'ca4227_out_noise': [],
                  'ca4227_rmax': [],
                  'fe4383_cen': [],
                  'fe4383_cen_noise': [],
                  'fe4383_mid': [],
                  'fe4383_mid_noise': [],
                  'fe4383_out': [],
                  'fe4383_out_noise': [],
                  'fe4383_rmax': [],
                  'fe5270_cen': [],
                  'fe5270_cen_noise': [],
                  'fe5270_mid': [],
                  'fe5270_mid_noise': [],
                  'fe5270_out': [],
                  'fe5270_out_noise': [],
                  'fe5270_rmax': [],
                  'cn1_cen': [],
                  'cn1_cen_noise': [],
                  'cn1_mid': [],
                  'cn1_mid_noise': [],
                  'cn1_out': [],
                  'cn1_out_noise': [],
                  'cn1_rmax': [],
                  'mgb_cen': [],
                  'mgb_cen_noise': [],
                  'mgb_mid': [],
                  'mgb_mid_noise': [],
                  'mgb_out': [],
                  'mgb_out_noise': [],
                  'mgb_rmax': [],
                  'dn4000_cen': [],
                  'dn4000_cen_noise': [],
                  'dn4000_mid': [],
                  'dn4000_mid_noise': [],
                  'dn4000_out': [],
                  'dn4000_out_noise': [],
                  'dn4000_rmax': [],
                  'n1_5199_cen': [],
                  'n1_5199_mid': [],
                  'n1_5199_out': [],
                  'ha_cen': [],
                  'ha_mid': [],
                  'ha_out': [],
                  'o3_4960_cen': [],
                  'o3_4960_mid': [],
                  'o3_4960_out': [],
                  'o3_5008_cen': [],
                  'o3_5008_mid': [],
                  'o3_5008_out': [],
                  'mgfe_cen': [],
                  'mgfe_cen_noise': [],
                  'mgfe_mid': [],
                  'mgfe_mid_noise': [],
                  'mgfe_out': [],
                  'mgfe_out_noise': [],
                  'mgbfe_cen': [],
                  'mgbfe_cen_noise': [],
                  'mgbfe_mid': [],
                  'mgbfe_mid_noise': [],
                  'mgbfe_out': [],
                  'mgbfe_out_noise': [],
                  'sigma_1kpc': [],
                  'sigma_re_8': []}
    total, counts = 0, 0

    '''
    discarded_counts = fits.getdata('sample_mask.fits', 1)
    rmax_manga = np.zeros_like(discarded_counts['cen<psf'])
    discarded_counts['cen<psf'] = np.zeros_like(discarded_counts['cen<psf'])
    discarded_counts['small_pixnum']=np.zeros_like(discarded_counts['cen<psf'])
    mask_m = np.zeros_like(discarded_counts['cen<psf'])

    _, _, idx_m = np.intersect1d(sga_new_ifu[sga_new_mass < 11.2],
                                 manga_sga[1].data['plateifu'], return_indices=True)
    mask_m[idx_m] = mask_m[idx_m] + 1

    mask_o = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_o = np.intersect1d(outliers, manga_sga[1].data['plateifu'], return_indices=True)
    mask_o[idx_o] = mask_o[idx_o] + 1

    mask_e = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_e = np.intersect1d(sga_new_ifu[sga_mass['ba'] < 0.3], manga_sga[1].data['plateifu'], return_indices=True)
    mask_e[idx_e] = mask_e[idx_e] + 1

    mask_nm = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_nm = np.intersect1d(outliers_mass, manga_sga[1].data['plateifu'], return_indices=True)
    mask_nm[idx_nm] = mask_nm[idx_nm] + 1

    mask_psf = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_psf = np.intersect1d(outliers_psf, manga_sga[1].data['plateifu'], return_indices=True)
    mask_psf[idx_psf] = mask_psf[idx_psf] + 1

    mask_visual = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_visual = np.intersect1d(outliers_visual, manga_sga[1].data['plateifu'], return_indices=True)
    mask_visual[idx_visual] = mask_visual[idx_visual] + 1

    mask_spec = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_spec = np.intersect1d(outliers_spec, manga_sga[1].data['plateifu'], return_indices=True)
    mask_spec[idx_spec] = mask_spec[idx_spec] + 1

    mask_reobs = np.zeros_like(discarded_counts['cen<psf'])
    _, _, idx_reobs = np.intersect1d(reobserved, manga_sga[1].data['plateifu'], return_indices=True)
    mask_reobs[idx_reobs] = mask_reobs[idx_reobs] + 1

    '''
    #'''
    mask_names = ['DEC_30',
                  'no_maps',
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
    for i in range(1, len(mask_names)):
        mask = mask & (mask_hdu[1].data[mask_names[i]] == 0)

    #'''
    # check = ['8085-9101', '8726-1901', '8726-1902', '8726-6103', '8934-12702']
    factor_decalz = {'g': [0.08804478, -0.01202266],
                     'r': [0.097115220, -0.005264246],
                     'z': [0.0504410920, -0.0002501634]}
    factor_bass = {'g': [0.011121011, -0.009446287],
                   'r': [0.080892829, -0.003455136],
                   'z': [0.073761688, 0.004807630]}

    '''
    factor_bass = {'g': [0., 0.],
                   'r': [0., -0.],
                   'z': [0., -0.]}
    factor_decalz = {'g': [0.038, -0.039],
                     'r': [0.035, -0.04],
                     'z': [-0.008, 0.05]}
    '''

    for gal in manga_sga[1].data[idx_total]:
        total += 1
        ifu = gal['plateifu']
        # if outliers_spec.__contains__(ifu):
        #    continue
        dec = gal['objdec']
        sid = gal['sgaid']
        idx_msk = np.where(mask_hdu[1].data['plateifu'] == ifu)[0][0]
        #'''
        if not mask[idx_msk]:
            continue
        #'''

        if not sga_mass['plateifu'].__contains__(ifu):
            print(ifu)
            continue
        idx_ml = np.where(sga_mass['plateifu'] == ifu)[0][0]
        ml = sga_mass['lg_M/L'][idx_ml]

        idx_sga = np.where(sga[1].data['sga_id'] == sid)[0][0]
        gn = sga[1].data['group_name'][idx_sga]
        z_sga = sga[1].data['z_leda'][idx_sga]
        sid = str(sid)
        file_path = '/home/xiaoya/sample_select/tmp/'
        maps_path = '/scratch/shared_data/manga/MPL11-DAP/SPX-MILESHC-MASTARSSP/' + ifu[0:ifu.find(
            '-')] + '/'
        if not os.path.exists(maps_path + 'manga-' + ifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz'):
            # print('no maps', ifu)
            maps_path = '/home/xiaoya/complimentary_maps/'
            # continue
        if os.stat(maps_path + 'manga-' + ifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz').st_size == 0:
            # print('zero maps', ifu)
            maps_path = '/home/xiaoya/complimentary_maps/'
            # continue
        # '''
        if not os.path.exists(file_path + gn + '-largegalaxy-' + sid + '-ellipse.fits'):
            # print('no sga ellip', ifu)
            continue
        # '''
        hdu = fits.open(file_path + gn + '-largegalaxy-' + sid + '-ellipse.fits')
        # '''

        if not isinstance(hdu[1].data['R_INTENS'][0], np.ndarray):
            # print('no sga intens', ifu)
            continue
        # '''

        hdap = fits.open(maps_path + 'manga-' + ifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz')
        d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value
        d_l = cosm.luminosity_distance(z_sga).to('pc').value
        rpsf = 2.5 * (d_a * np.pi / (3600 * 180))
        reff = sga_mass['Re_kpc'][idx_ml]
        r_center = reff / 2

        '''
        if r_center < rpsf:
            # print('cen<rpsf', ifu)
            discarded_counts['cen<psf'][idx_msk] += 1
            continue
        '''
        ebv = hdap[0].header['EBVGAL']
        ba = sga_mass['ba'][idx_ml]  # 1 - hdu[1].data['R_EPS'][0][1]
        ag = sga_mass['pa'][idx_ml]  # hdu[1].data['R_PA'][0][1]
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
        # reff = sga_mass['Re_arc'][idx_ml]#*2
        A = A * (d_a * np.pi / (3600 * 180))

        idx_cen = np.where(hdap['SPX_ELLCOO'].data[0] == np.nanmin(hdap['SPX_ELLCOO'].data[0]))
        yc, xc = idx_cen[0][0], idx_cen[1][0]

        # A[~(hdap['SPX_SNR'].data > 3)] = np.nan
        snr_mask = {'g': hdu[1].data['G_INTENS'][0] / hdu[1].data['G_INTENS_ERR'][0] > 3,
                    'r': hdu[1].data['R_INTENS'][0] / hdu[1].data['R_INTENS_ERR'][0] > 3,
                    'z': hdu[1].data['Z_INTENS'][0] / hdu[1].data['Z_INTENS_ERR'][0] > 3}

        sga_grz_original = {'g': get_cog(hdu[1].data['G_SMA'][0], hdu[1].data['G_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['G_EPS'][0]) * 0.263 ** 2,
                            'r': get_cog(hdu[1].data['R_SMA'][0], hdu[1].data['R_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['R_EPS'][0]) * 0.263 ** 2,
                            'z': get_cog(hdu[1].data['Z_SMA'][0], hdu[1].data['Z_INTENS'][0],
                                         axis_ratio=1 - hdu[1].data['Z_EPS'][0]) * 0.263 ** 2}
        lummax_0 = {}
        lum20_0 = {}
        lum10_0 = {}
        lumcen_0 = {}
        lumout_re_0 = {}
        lum30_0 = {}
        lumout20_0 = {}
        lumout_0 = {}
        A[np.isnan(hdap['SPX_SNR'].data)] = np.nan

        mask_snr = hdap['SPX_SNR'].data > 2

        bins_in = A <= reff * 0.5  # 0.4#5 / d_u  # reff / 2
        bins_out = A >= 1 * reff  # 1.1*reff#15/d_u#reff
        bins_mid = (A >= reff * 0.5) & (A <= reff)  # (A >= 5 / d_u) & (A <= 15/d_u)#(A >= reff / 2) & (A <= reff)
        bining = np.full_like(bins_in * 1, -1)
        bining[bins_in] = 1
        bining[bins_mid] = 2
        bining[bins_out] = 3

        # for num_bin in range(8):
        #    bining[bins_r[num_bin]] = num_bin+1
        bining[~(hdap['SPX_SNR'].data >= 2)] = -1
        bining[hdap['STELLAR_VEL_MASK'].data >= int(2 ** 30)] = -1

        mask_out = (bining == 3)
        mask_mid = (bining == 2)
        mask_cen = (bining == 1)
        mask_1kpc = A <= 1
        mask_re8 = A <= reff / 8

        if masking['info'].data['plateifu'].__contains__(ifu):
            mask_snr = mask_snr & (masking[ifu].data == 0)
        '''
        if np.nansum(((bining==1) & mask_snr) * 1) < 20 or np.nansum(((bining==2) & mask_snr) * 1) < 20 or np.nansum(((bining==3) & mask_snr) * 1) < 20:
            discarded_counts['small_pixnum'][idx_msk] += 1
        '''
        rmax_manga[idx_msk] = np.nanmax(A[bining > -1])
        # print('small pixnum', ifu)
        # continue
        # cens.append(r_center)
        # reffs.append(reff)
        reff = sga_mass['Re_kpc'][idx_ml]

        for band in ['g', 'r', 'z']:
            dx = hdu[1].data[band + '_SMA'][0] * 0.263 * (d_a * np.pi / (3600 * 180))
            dx[dx < 1] = np.nan
            # dx = dx / reff
            dx = dx[snr_mask[band]]
            dy = sga_grz_original[band][snr_mask[band]]
            dy = dy[~np.isnan(dx)]
            dx = dx[~np.isnan(dx)]
            diff = np.insert(abs(np.diff(dy)), 0, 0)
            idx_diff0 = np.where((diff > np.nanstd(dy[(dx > reff) & (dx < 1.5 * reff)])) & (dx > 1.5 * reff))[0]
            if len(idx_diff0) >= 1:
                dy[idx_diff0[0]:] = np.nan
            if np.nanmax(dx) <= 30:
                lummax_0[band] = np.nan
                lumcen_0[band] = np.nan
                lumout_re_0[band] = np.nan
                lum30_0[band] = np.nan
                lum20_0[band] = np.nan
                lum10_0[band] = np.nan
                lumout20_0[band] = np.nan
            else:
                lummax_0[band] = np.nanmax(dy)
                # dy[dy > 1.1 * np.nanmax(dy[dx <= 1])] = np.nan
                func = interpolate.interp1d(dx, dy)
                lum30_0[band] = func(30)
                lum20_0[band] = func(20)
                lum10_0[band] = func(20)

                if np.nanmin(dx) > r_center:
                    lumcen_0[band] = np.nan
                else:
                    lumcen_0[band] = func(r_center)
                # lumout_re_0[band] = func(np.nanmin([np.nanmax(dx), np.nanmax(A)])) - func(
                #    np.nanmin([np.nanmax(dx), reff]))
                lumout_re_0[band] = func(np.nanmax(dx)) - func(np.nanmin([np.nanmax(dx), reff]))
                r_max[0].append(np.nanmax(dx))
                r_max[1].append(np.nanmax(A[hdap['SPX_SNR'].data > 2]))
            lumout_0[band] = lummax_0[band] - lum30_0[band]
            lumout20_0[band] = lummax_0[band] - lum20_0[band]

            lummax_0[band] = 22.5 - 2.5 * np.log10(lummax_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lum20_0[band] = 22.5 - 2.5 * np.log10(lum20_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lum10_0[band] = 22.5 - 2.5 * np.log10(lum10_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lumout_0[band] = 22.5 - 2.5 * np.log10(lumout_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lumout20_0[band] = 22.5 - 2.5 * np.log10(lumout20_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lumcen_0[band] = 22.5 - 2.5 * np.log10(lumcen_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
            lumout_re_0[band] = 22.5 - 2.5 * np.log10(lumout_re_0[band]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10)
        if dec < 32.375:
            lumout_1 = {
                'g': lumout_0['g'] + factor_decalz['g'][0] * (lumout_0['g'] - lumout_0['r']) + factor_decalz['g'][1],
                'r': lumout_0['r'] + factor_decalz['r'][0] * (lumout_0['r'] - lumout_0['z']) + factor_decalz['r'][1],
                'z': lumout_0['z'] + factor_decalz['z'][0] * (lumout_0['r'] - lumout_0['z']) + factor_decalz['z'][1]}
            lumout20_1 = {
                'g': lumout20_0['g'] + factor_decalz['g'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_decalz['g'][
                    1],
                'r': lumout20_0['r'] + factor_decalz['r'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_decalz['r'][
                    1],
                'z': lumout20_0['z'] + factor_decalz['z'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_decalz['z'][
                    1]}
            lum20_1 = {
                'g': lum20_0['g'] + factor_decalz['g'][0] * (lum20_0['g'] - lum20_0['r']) + factor_decalz['g'][1],
                'r': lum20_0['r'] + factor_decalz['r'][0] * (lum20_0['r'] - lum20_0['z']) + factor_decalz['r'][1],
                'z': lum20_0['z'] + factor_decalz['z'][0] * (lum20_0['r'] - lum20_0['z']) + factor_decalz['z'][1]}
            lum10_1 = {
                'g': lum10_0['g'] + factor_decalz['g'][0] * (lum10_0['g'] - lum10_0['r']) + factor_decalz['g'][1],
                'r': lum10_0['r'] + factor_decalz['r'][0] * (lum10_0['r'] - lum10_0['z']) + factor_decalz['r'][1],
                'z': lum10_0['z'] + factor_decalz['z'][0] * (lum10_0['r'] - lum10_0['z']) + factor_decalz['z'][1]}
            lummax_1 = {
                'g': lummax_0['g'] + factor_decalz['g'][0] * (lummax_0['g'] - lummax_0['r']) + factor_decalz['g'][1],
                'r': lummax_0['r'] + factor_decalz['r'][0] * (lummax_0['r'] - lummax_0['z']) + factor_decalz['r'][1],
                'z': lummax_0['z'] + factor_decalz['z'][0] * (lummax_0['r'] - lummax_0['z']) + factor_decalz['z'][1]}
            lumcen_1 = {
                'g': lumcen_0['g'] + factor_decalz['g'][0] * (lumcen_0['g'] - lumcen_0['r']) + factor_decalz['g'][1],
                'r': lumcen_0['r'] + factor_decalz['r'][0] * (lumcen_0['r'] - lumcen_0['z']) + factor_decalz['r'][1],
                'z': lumcen_0['z'] + factor_decalz['z'][0] * (lumcen_0['r'] - lumcen_0['z']) + factor_decalz['z'][1]}
            # k-correction: http://kcor.sai.msu.ru/
            lumout_re_1 = {
                'g': lumout_re_0['g'] + factor_decalz['g'][0] * (lumout_re_0['g'] - lumout_re_0['r']) +
                     factor_decalz['g'][1],
                'r': lumout_re_0['r'] + factor_decalz['r'][0] * (lumout_re_0['r'] - lumout_re_0['z']) +
                     factor_decalz['r'][1],
                'z': lumout_re_0['z'] + factor_decalz['z'][0] * (lumout_re_0['r'] - lumout_re_0['z']) +
                     factor_decalz['z'][1]}
            # k-correction: http://kcor.sai.msu.ru/
        else:
            lumout_1 = {
                'g': lumout_0['g'] + factor_bass['g'][0] * (lumout_0['g'] - lumout_0['r']) + factor_bass['g'][1],
                'r': lumout_0['r'] + factor_bass['r'][0] * (lumout_0['g'] - lumout_0['r']) + factor_bass['r'][1],
                'z': lumout_0['z'] + factor_bass['z'][0] * (lumout_0['r'] - lumout_0['z']) + factor_bass['z'][1]}
            lumout20_1 = {
                'g': lumout20_0['g'] + factor_bass['g'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_bass['g'][1],
                'r': lumout20_0['r'] + factor_bass['r'][0] * (lumout20_0['g'] - lumout20_0['r']) + factor_bass['r'][1],
                'z': lumout20_0['z'] + factor_bass['z'][0] * (lumout20_0['r'] - lumout20_0['z']) + factor_bass['z'][1]}
            lum20_1 = {
                'g': lum20_0['g'] + factor_bass['g'][0] * (lum20_0['g'] - lum20_0['r']) + factor_bass['g'][1],
                'r': lum20_0['r'] + factor_bass['r'][0] * (lum20_0['g'] - lum20_0['r']) + factor_bass['r'][1],
                'z': lum20_0['z'] + factor_bass['z'][0] * (lum20_0['r'] - lum20_0['z']) + factor_bass['z'][1]}
            lum10_1 = {
                'g': lum10_0['g'] + factor_bass['g'][0] * (lum10_0['g'] - lum10_0['r']) + factor_bass['g'][1],
                'r': lum10_0['r'] + factor_bass['r'][0] * (lum10_0['g'] - lum10_0['r']) + factor_bass['r'][1],
                'z': lum10_0['z'] + factor_bass['z'][0] * (lum10_0['r'] - lum10_0['z']) + factor_bass['z'][1]}
            lummax_1 = {
                'g': lummax_0['g'] + factor_bass['g'][0] * (lummax_0['g'] - lummax_0['r']) + factor_bass['g'][1],
                'r': lummax_0['r'] + factor_bass['r'][0] * (lummax_0['g'] - lummax_0['r']) + factor_bass['r'][1],
                'z': lummax_0['z'] + factor_bass['z'][0] * (lummax_0['r'] - lummax_0['z']) + factor_bass['z'][1]}
            lumcen_1 = {
                'g': lumcen_0['g'] + factor_bass['g'][0] * (lumcen_0['g'] - lumcen_0['r']) + factor_bass['g'][1],
                'r': lumcen_0['r'] + factor_bass['r'][0] * (lumcen_0['g'] - lumcen_0['r']) + factor_bass['r'][1],
                'z': lumcen_0['z'] + factor_bass['z'][0] * (lumcen_0['r'] - lumcen_0['z']) + factor_bass['z'][1]}
            # k-correction: http://kcor.sai.msu.ru/
            lumout_re_1 = {
                'g': lumout_re_0['g'] + factor_bass['g'][0] * (lumout_re_0['g'] - lumout_re_0['r']) + factor_bass['g'][
                    1],
                'r': lumout_re_0['r'] + factor_bass['r'][0] * (lumout_re_0['g'] - lumout_re_0['r']) + factor_bass['r'][
                    1],
                'z': lumout_re_0['z'] + factor_bass['z'][0] * (lumout_re_0['r'] - lumout_re_0['z']) + factor_bass['z'][
                    1]}
            # k-correction: http://kcor.sai.msu.ru/

        # k-correction: http://kcor.sai.msu.ru/
        lumout = {
            'g': lumout_1['g'] - calc_kcor('g', z_sga, 'gr', lumout_1['g'] - lumout_1['r']),
            'r': lumout_1['r'] - calc_kcor('r', z_sga, 'gr', lumout_1['g'] - lumout_1['r']),
            'z': lumout_1['r'] - calc_kcor('z', z_sga, 'gz', lumout_1['g'] - lumout_1['z'])
        }
        lumout20 = {
            'g': lumout20_1['g'] - calc_kcor('g', z_sga, 'gr', lumout20_1['g'] - lumout20_1['r']),
            'r': lumout20_1['r'] - calc_kcor('r', z_sga, 'gr', lumout20_1['g'] - lumout20_1['r']),
            'z': lumout20_1['r'] - calc_kcor('z', z_sga, 'gz', lumout20_1['g'] - lumout20_1['z'])
        }

        # k-correction: http://kcor.sai.msu.ru/
        lum20 = {
            'g': lum20_1['g'] - calc_kcor('g', z_sga, 'gr', lum20_1['g'] - lum20_1['r']),
            'r': lum20_1['r'] - calc_kcor('r', z_sga, 'gr', lum20_1['g'] - lum20_1['r']),
            'z': lum20_1['r'] - calc_kcor('z', z_sga, 'gz', lum20_1['g'] - lum20_1['z'])
        }
        # k-correction: http://kcor.sai.msu.ru/
        lum10 = {
            'g': lum10_1['g'] - calc_kcor('g', z_sga, 'gr', lum10_1['g'] - lum10_1['r']),
            'r': lum10_1['r'] - calc_kcor('r', z_sga, 'gr', lum10_1['g'] - lum10_1['r']),
            'z': lum10_1['r'] - calc_kcor('z', z_sga, 'gz', lum10_1['g'] - lum10_1['z'])
        }

        # k-correction: http://kcor.sai.msu.ru/
        lummax = {
            'g': lummax_1['g'] - calc_kcor('g', z_sga, 'gr', lummax_1['g'] - lummax_1['r']),
            'r': lummax_1['r'] - calc_kcor('r', z_sga, 'gr', lummax_1['g'] - lummax_1['r']),
            'z': lummax_1['r'] - calc_kcor('z', z_sga, 'gz', lummax_1['g'] - lummax_1['z'])
        }
        lumcen = {
            'g': lumcen_1['g'] - calc_kcor('g', z_sga, 'gr', lumcen_1['g'] - lumcen_1['r']),
            'r': lumcen_1['r'] - calc_kcor('r', z_sga, 'gr', lumcen_1['g'] - lumcen_1['r']),
            'z': lumcen_1['r'] - calc_kcor('z', z_sga, 'gz', lumcen_1['g'] - lumcen_1['z'])
        }
        lumout_re = {
            'g': lumout_re_1['g'] - calc_kcor('g', z_sga, 'gr', lumout_re_1['g'] - lumout_re_1['r']),
            'r': lumout_re_1['r'] - calc_kcor('r', z_sga, 'gr', lumout_re_1['g'] - lumout_re_1['r']),
            'z': lumout_re_1['r'] - calc_kcor('z', z_sga, 'gz', lumout_re_1['g'] - lumout_re_1['z'])
        }

        for band in ['g', 'r', 'z']:
            lum20[band] = -0.4 * (lum20[band] - solar_l[band])
            lum10[band] = -0.4 * (lum10[band] - solar_l[band])
            lumout[band] = -0.4 * (lumout[band] - solar_l[band])
            lumout20[band] = -0.4 * (lumout20[band] - solar_l[band])

            lummax[band] = -0.4 * (lummax[band] - solar_l[band])
            lumcen[band] = -0.4 * (lumcen[band] - solar_l[band])
            lumout_re[band] = -0.4 * (lumout_re[band] - solar_l[band])

        # lum_dist.append(np.array([lum20['r'], lumout['r']]))

        ha = hdap["EMLINE_SFLUX"].data[23]
        mask1 = (~(hdap["EMLINE_SFLUX"].data[23] * np.sqrt(hdap["EMLINE_SFLUX_IVAR"].data[23]) > 1)) | (
            ~(hdap["EMLINE_SFLUX"].data[23] > 0)) | (np.log2(hdap['EMLINE_SFLUX_MASK'].data[23]) >= 30) | (
                        np.log2(hdap['EMLINE_SFLUX_MASK'].data[23]) == 8)
        ha[mask1] = np.nan
        hb = hdap["EMLINE_SFLUX"].data[14]
        mask2 = (~(hdap["EMLINE_SFLUX"].data[14] * np.sqrt(hdap["EMLINE_SFLUX_IVAR"].data[14]) > 1)) | (
            ~(hdap["EMLINE_SFLUX"].data[14] > 0)) | (np.log2(hdap['EMLINE_SFLUX_MASK'].data[14]) >= 30) | (
                        np.log2(hdap['EMLINE_SFLUX_MASK'].data[14]) == 8)
        hb[mask2] = np.nan
        ebv_g = 0.935 * np.log((ha / hb) / 2.88)
        lam = [4960.295, 5008.240, 5199.349, 6564.608]
        atn = [cal_gas(lam[k], ha, hb) for k in range(len(lam))]
        ha = np.log10(ha) + np.log10(emlines_sb(z_sga)) + atn[-1]

        n1_5199 = np.log10(hdap['EMLINE_SFLUX'].data[17]) + np.log10(emlines_sb(z_sga)) + atn[-2]
        n1_5199_noise = 1 / np.sqrt(hdap['EMLINE_SFLUX_IVAR'].data[12])
        mask_tmp = (~(n1_5199 / n1_5199_noise > 1)) | (np.log2(hdap['EMLINE_SFLUX_MASK'].data[17]) >= 30) | (
                np.log2(hdap['EMLINE_SFLUX_MASK'].data[17]) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        n1_5199[mask_tmp] = np.nan

        o3_4960 = np.log10(hdap['EMLINE_SFLUX'].data[15]) + np.log10(emlines_sb(z_sga)) + atn[0]
        o3_4960_noise = 1 / np.sqrt(hdap['EMLINE_SFLUX_IVAR'].data[15])
        mask_tmp = (~(o3_4960 / o3_4960_noise > 1)) | (np.log2(hdap['EMLINE_SFLUX_MASK'].data[15]) >= 30) | (
                np.log2(hdap['EMLINE_SFLUX_MASK'].data[15]) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        o3_4960[mask_tmp] = np.nan

        o3_5008 = np.log10(hdap['EMLINE_SFLUX'].data[16]) + np.log10(emlines_sb(z_sga)) + atn[1]
        o3_5008_noise = 1 / np.sqrt(hdap['EMLINE_SFLUX_IVAR'].data[16])
        mask_tmp = (~(o3_5008 / o3_5008_noise > 1)) | (np.log2(hdap['EMLINE_SFLUX_MASK'].data[16]) >= 30) | (
                np.log2(hdap['EMLINE_SFLUX_MASK'].data[16]) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        o3_5008[mask_tmp] = np.nan

        sigma = np.sqrt(hdap['STELLAR_SIGMA'].data ** 2 - hdap['STELLAR_SIGMACORR'].data[0, :, :] ** 2)
        sigma_noise = 1 / np.sqrt(hdap['STELLAR_SIGMA_IVAR'].data)
        mask_tmp = (~(sigma / sigma_noise > 1)) | (np.log2(hdap['STELLAR_SIGMA_MASK'].data) >= 30) | (
                np.log2(hdap['STELLAR_SIGMA_MASK'].data) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        sigma[mask_tmp] = np.nan
        sigma_noise[mask_tmp] = np.nan
        sigma_rmax = np.nanmax(A[~np.isnan(sigma)]) / reff

        cn1 = hdap['SPECINDEX'].data[0] + hdap['SPECINDEX_CORR'].data[0]
        cn1_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[0])
        mask_tmp = (np.log2(hdap['SPECINDEX_MASK'].data[0]) >= 30) | (np.log2(hdap['SPECINDEX_MASK'].data[0]) == 8)
        if np.nansum((~mask_tmp) * 1) <= 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        cn1[mask_tmp] = np.nan
        cn1_noise[mask_tmp] = np.nan
        cn1_rmax = np.nanmax(A[~np.isnan(cn1)]) / reff

        dn4000 = hdap['SPECINDEX'].data[44] * hdap['SPECINDEX_CORR'].data[44]
        dn4000_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[44])
        mask_tmp = (~(dn4000 / dn4000_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[44]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[44]) == 8)
        if np.nansum((~mask_tmp) * 1) < 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        dn4000[mask_tmp] = np.nan
        dn4000_noise[mask_tmp] = np.nan
        dn4000_rmax = np.nanmax(A[~np.isnan(dn4000)]) / reff

        ca4227 = hdap['SPECINDEX'].data[2] * hdap['SPECINDEX_CORR'].data[2]
        ca4227_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[2])
        mask_tmp = (~(ca4227 / ca4227_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[2]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[2]) == 8)
        if np.nansum((~mask_tmp) * 1) < 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        ca4227[mask_tmp] = np.nan
        ca4227_noise[mask_tmp] = np.nan
        ca4227_rmax = np.nanmax(A[~np.isnan(ca4227)]) / reff

        fe4383 = hdap['SPECINDEX'].data[4] * hdap['SPECINDEX_CORR'].data[4]
        fe4383_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[4])
        mask_tmp = (~(fe4383 / fe4383_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[4]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[4]) == 8)
        if np.nansum((~mask_tmp) * 1) < 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        fe4383[mask_tmp] = np.nan
        fe4383_noise[mask_tmp] = np.nan
        fe4383_rmax = np.nanmax(A[~np.isnan(fe4383)]) / reff

        fe5270 = hdap['SPECINDEX'].data[13] * hdap['SPECINDEX_CORR'].data[13]
        fe5270_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[13])
        mask_tmp = (~(fe5270 / fe5270_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[13]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[13]) == 8)
        if np.nansum((~mask_tmp) * 1) < 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        fe5270[mask_tmp] = np.nan
        fe5270_noise[mask_tmp] = np.nan
        fe5270_rmax = np.nanmax(A[~np.isnan(fe5270)]) / reff

        fe5335 = hdap['SPECINDEX'].data[14] * hdap['SPECINDEX_CORR'].data[14]
        fe5335_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[14])
        mask_tmp = (~(fe5335 / fe5335_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[14]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[14]) == 8)
        fe5335[mask_tmp] = np.nan
        fe5335_noise[mask_tmp] = np.nan

        mgb = hdap['SPECINDEX'].data[12] * hdap['SPECINDEX_CORR'].data[12]
        mgb_noise = 1 / np.sqrt(hdap['SPECINDEX_IVAR'].data[12])
        mask_tmp = (~(mgb / mgb_noise > 1)) | (np.log2(hdap['SPECINDEX_MASK'].data[12]) >= 30) | (
                np.log2(hdap['SPECINDEX_MASK'].data[12]) == 8)
        if np.nansum((~mask_tmp) * 1) < 1:
            print('mask shit', ifu)
            mask_tmp[int(D / 2) - 1:int(D / 2) + 1, int(D / 2) - 1:int(D / 2) + 1] = False
        mgb[mask_tmp] = np.nan
        mgb_noise[mask_tmp] = np.nan
        mgb_rmax = np.nanmax(A[~np.isnan(mgb)]) / reff

        mgfe = np.sqrt(mgb * (0.72 * fe5270 + 0.28 * fe5335))
        mgfe_noise = 0.25 * (mgb_noise ** 2 * (0.72 * fe5270 + 0.28 * fe5335) ** 2 +
                             (fe5270_noise * 0.72 * mgfe) ** 2 +
                             (fe5335_noise * 0.28 * mgfe) ** 2) / mgfe ** 2
        mgfe[~(mgfe > (mgfe_noise * 1))] = np.nan
        mgfe_noise[~(mgfe > (mgfe_noise * 1))] = np.nan

        mgbfe = mgb / (0.5 * (fe5270 + fe5335))
        mgbfe_noise = 2 * np.sqrt(mgb_noise ** 2 * (fe5270 + fe5335) ** 2 +
                                  mgb ** 2 * (fe5335_noise ** 2 + fe5270_noise ** 2)) / (fe5270 + fe5335) ** 2
        mgbfe[~(mgbfe > (mgbfe_noise * 1))] = np.nan
        mgbfe_noise[~(mgbfe > (mgbfe_noise * 1))] = np.nan

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
        # '''

        parameters['fe5270_rmax'].append(fe5270_rmax)
        parameters['mgb_rmax'].append(mgb_rmax)
        parameters['fe4383_rmax'].append(fe4383_rmax)
        parameters['ca4227_rmax'].append(ca4227_rmax)
        parameters['dn4000_rmax'].append(dn4000_rmax)
        parameters['cn1_rmax'].append(cn1_rmax)
        parameters['sigma_rmax'].append(sigma_rmax)

        parameters['taken_cen'].append(np.nansum((mask_cen & mask_snr) * 1))
        parameters['taken_mid'].append(np.nansum((mask_mid & mask_snr) * 1))
        parameters['taken_out'].append(np.nansum((mask_out & mask_snr) * 1))

        parameters['all_cen'].append(np.nansum(1 * mask_cen))
        parameters['all_mid'].append(np.nansum(1 * mask_mid))
        parameters['all_out'].append(np.nansum(1 * mask_out))

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

        parameters['cn1_cen'].append(np.nanmean(cn1[mask_cen == 1]))
        parameters['cn1_cen_noise'].append(
            np.sqrt(np.nansum(cn1_noise[mask_cen == 1] ** 2)) / np.nansum(1 * (~np.isnan(cn1_noise[mask_cen == 1]))))
        parameters['cn1_mid'].append(np.nanmean(cn1[mask_mid == 1]))
        parameters['cn1_mid_noise'].append(
            np.sqrt(np.nansum(cn1_noise[mask_mid == 1] ** 2)) / np.nansum(1 * (~np.isnan(cn1_noise[mask_mid == 1]))))
        parameters['cn1_out'].append(np.nanmean(cn1[mask_out == 1]))
        parameters['cn1_out_noise'].append(
            np.sqrt(np.nansum(cn1_noise[mask_out == 1] ** 2)) / np.nansum(1 * (~np.isnan(cn1_noise[mask_out == 1]))))

        parameters['ca4227_cen'].append(np.nanmean(ca4227[mask_cen == 1]))
        parameters['ca4227_cen_noise'].append(np.sqrt(np.nansum(ca4227_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(ca4227_noise[mask_cen == 1]))))
        parameters['ca4227_mid'].append(np.nanmean(ca4227[mask_mid == 1]))
        parameters['ca4227_mid_noise'].append(np.sqrt(np.nansum(ca4227_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(ca4227_noise[mask_mid == 1]))))
        parameters['ca4227_out'].append(np.nanmean(ca4227[mask_out == 1]))
        parameters['ca4227_out_noise'].append(np.sqrt(np.nansum(ca4227_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(ca4227_noise[mask_out == 1]))))

        parameters['fe4383_cen'].append(np.nanmean(fe4383[mask_cen == 1]))
        parameters['fe4383_cen_noise'].append(np.sqrt(np.nansum(fe4383_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe4383_noise[mask_cen == 1]))))
        parameters['fe4383_mid'].append(np.nanmean(fe4383[mask_mid == 1]))
        parameters['fe4383_mid_noise'].append(np.sqrt(np.nansum(fe4383_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe4383_noise[mask_mid]))))
        parameters['fe4383_out'].append(np.nanmean(fe4383[mask_out == 1]))
        parameters['fe4383_out_noise'].append(np.sqrt(np.nansum(fe4383_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe4383_noise[mask_out == 1]))))

        parameters['fe5270_cen'].append(np.nanmean(fe5270[mask_cen == 1]))
        parameters['fe5270_cen_noise'].append(np.sqrt(np.nansum(fe5270_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5270_noise[mask_cen == 1]))))
        parameters['fe5270_mid'].append(np.nanmean(fe5270[mask_mid == 1]))
        parameters['fe5270_mid_noise'].append(np.sqrt(np.nansum(fe5270_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5270_noise[mask_mid == 1]))))
        parameters['fe5270_out'].append(np.nanmean(fe5270[mask_out == 1]))
        parameters['fe5270_out_noise'].append(np.sqrt(np.nansum(fe5270_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5270_noise[mask_out == 1]))))

        parameters['fe5335_cen'].append(np.nanmean(fe5335[mask_cen == 1]))
        parameters['fe5335_cen_noise'].append(np.sqrt(np.nansum(fe5335_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5335_noise[mask_cen == 1]))))
        parameters['fe5335_mid'].append(np.nanmean(fe5335[mask_mid == 1]))
        parameters['fe5335_mid_noise'].append(np.sqrt(np.nansum(fe5335_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5335_noise[mask_mid == 1]))))
        parameters['fe5335_out'].append(np.nanmean(fe5335[mask_out == 1]))
        parameters['fe5335_out_noise'].append(np.sqrt(np.nansum(fe5335_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(fe5335_noise[mask_out == 1]))))



        parameters['mgb_cen'].append(np.nanmean(mgb[mask_cen == 1]))
        parameters['mgb_cen_noise'].append(
            np.sqrt(np.nansum(mgb_noise[mask_cen == 1] ** 2)) / np.nansum(1 * (~np.isnan(mgb_noise[mask_cen == 1]))))
        parameters['mgb_mid'].append(np.nanmean(mgb[mask_mid == 1]))
        parameters['mgb_mid_noise'].append(np.sqrt(np.nansum(mgb_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(mgb_noise[mask_mid == 1]))))
        parameters['mgb_out'].append(np.nanmean(mgb[mask_out == 1]))
        parameters['mgb_out_noise'].append(
            np.sqrt(np.nansum(mgb_noise[mask_out == 1] ** 2)) / np.nansum(1 * (~np.isnan(mgb_noise[mask_out == 1]))))

        parameters['dn4000_cen'].append(np.nanmean(dn4000[mask_cen == 1]))
        parameters['dn4000_cen_noise'].append(np.sqrt(np.nansum(dn4000_noise[mask_cen == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(dn4000_noise[mask_cen == 1]))))
        parameters['dn4000_mid'].append(np.nanmean(dn4000[mask_mid == 1]))
        parameters['dn4000_mid_noise'].append(np.sqrt(np.nansum(dn4000_noise[mask_mid == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(dn4000_noise[mask_mid == 1]))))
        parameters['dn4000_out'].append(np.nanmean(dn4000[mask_out == 1]))
        parameters['dn4000_out_noise'].append(np.sqrt(np.nansum(dn4000_noise[mask_out == 1] ** 2)) / np.nansum(
            1 * (~np.isnan(dn4000_noise[mask_out == 1]))))

        parameters['mgfe_cen'].append(np.nanmean(mgfe[mask_cen == 1]))
        parameters['mgfe_cen_noise'].append(
            np.sqrt(np.nansum(mgfe_noise[mask_cen == 1] ** 2)) / np.nansum(1 * (~np.isnan(mgfe_noise[mask_cen == 1]))))
        parameters['mgfe_mid'].append(np.nanmean(mgfe[mask_mid == 1]))
        parameters['mgfe_mid_noise'].append(
            np.sqrt(np.nansum(mgfe_noise[mask_mid == 1] ** 2)) / np.nansum(1 * (~np.isnan(mgfe_noise[mask_mid == 1]))))
        parameters['mgfe_out'].append(np.nanmean(mgfe[mask_out == 1]))
        parameters['mgfe_out_noise'].append(
            np.sqrt(np.nansum(mgfe_noise[mask_out == 1] ** 2)) / np.nansum(1 * (~np.isnan(mgfe_noise[mask_out == 1]))))

        parameters['mgbfe_cen'].append(np.nanmean(mgbfe[mask_cen == 1]))
        parameters['mgbfe_cen_noise'].append(
            np.sqrt(np.nansum(mgbfe_noise[mask_cen == 1] ** 2)) / np.nansum(
                1 * (~np.isnan(mgbfe_noise[mask_cen == 1]))))
        parameters['mgbfe_mid'].append(np.nanmean(mgbfe[mask_mid == 1]))
        parameters['mgbfe_mid_noise'].append(
            np.sqrt(np.nansum(mgbfe_noise[mask_mid == 1] ** 2)) / np.nansum(
                1 * (~np.isnan(mgbfe_noise[mask_mid == 1]))))
        parameters['mgbfe_out'].append(np.nanmean(mgbfe[mask_out == 1]))
        parameters['mgbfe_out_noise'].append(
            np.sqrt(np.nansum(mgbfe_noise[mask_out == 1] ** 2)) / np.nansum(
                1 * (~np.isnan(mgbfe_noise[mask_out == 1]))))

        parameters['n1_5199_cen'].append(np.nanmean(n1_5199[mask_cen == 1]))
        parameters['n1_5199_mid'].append(np.nanmean(n1_5199[mask_mid == 1]))
        parameters['n1_5199_out'].append(np.nanmean(n1_5199[mask_out == 1]))

        parameters['ha_cen'].append(np.nanmean(ha[mask_cen == 1]))
        parameters['ha_mid'].append(np.nanmean(ha[mask_mid == 1]))
        parameters['ha_out'].append(np.nanmean(ha[mask_out == 1]))

        parameters['o3_4960_cen'].append(np.nanmean(o3_4960[mask_cen == 1]))
        parameters['o3_4960_mid'].append(np.nanmean(o3_4960[mask_mid == 1]))
        parameters['o3_4960_out'].append(np.nanmean(o3_4960[mask_out == 1]))

        parameters['o3_5008_cen'].append(np.nanmean(o3_5008[mask_cen == 1]))
        parameters['o3_5008_mid'].append(np.nanmean(o3_5008[mask_mid == 1]))
        parameters['o3_5008_out'].append(np.nanmean(o3_5008[mask_out == 1]))

        parameters['snr_cen'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_cen == 1)] ** 2)))
        parameters['snr_mid'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_mid == 1)] ** 2)))
        parameters['snr_out'].append(np.sqrt(np.nansum(hdap['SPX_SNR'].data[(mask_out == 1)] ** 2)))

        parameters['lumout(max-30kpc)'].append(lumout['r'])
        parameters['lumout(max-20kpc)'].append(lumout20['r'])
        parameters['lum10kpc'].append(lum10['r'])
        parameters['lum20kpc'].append(lum20['r'])
        parameters['lummax'].append(lummax['r'])
        parameters['lum_cen'].append(lumcen['r'])
        parameters['lumout_re'].append(lumout_re['r'])

        parameters['massout(max-30kpc)'].append(lumout['r'] + ml)
        parameters['massout(max-20kpc)'].append(lumout20['r'] + ml)

        parameters['mass20kpc'].append(lum20['r'] + ml)
        parameters['mass10kpc'].append(lum10['r'] + ml)

        parameters['massmax'].append(lummax['r'] + ml)
        parameters['mass_cen'].append(lumcen['r'] + ml)
        parameters['massout_re'].append(lumout_re['r'] + ml)

        parameters['rmanga'].append(np.nanmax(A[(hdap['SPX_SNR'].data > 3)] / reff))
    t = Table([parameters['ifu'], parameters['ttype'], parameters['gzoo'],
               parameters['lummax'], parameters['lumout(max-30kpc)'], parameters['lum20kpc'],
               parameters['lum_cen'], parameters['lumout_re'], parameters['lumout(max-20kpc)'], parameters['lum10kpc'],
               parameters['massmax'], parameters['massout(max-30kpc)'], parameters['mass20kpc'],
               parameters['mass_cen'], parameters['massout_re'], parameters['massout(max-20kpc)'],
               parameters['mass10kpc'],
               parameters['rmanga'],
               parameters['snr_cen'], parameters['snr_mid'], parameters['snr_out'],
               parameters['all_cen'], parameters['all_mid'], parameters['all_out'],
               parameters['taken_cen'], parameters['taken_mid'], parameters['taken_out'],
               parameters['sigma_cen'], parameters['sigma_cen_noise'],
               parameters['sigma_mid'], parameters['sigma_mid_noise'],
               parameters['sigma_out'], parameters['sigma_out_noise'], parameters['sigma_rmax'],
               parameters['cn1_cen'], parameters['cn1_cen_noise'],
               parameters['cn1_mid'], parameters['cn1_mid_noise'],
               parameters['cn1_out'], parameters['cn1_out_noise'], parameters['cn1_rmax'],
               parameters['ca4227_cen'], parameters['ca4227_cen_noise'],
               parameters['ca4227_mid'], parameters['ca4227_mid_noise'],
               parameters['ca4227_out'], parameters['ca4227_out_noise'], parameters['ca4227_rmax'],
               parameters['fe4383_cen'], parameters['fe4383_cen_noise'],
               parameters['fe4383_mid'], parameters['fe4383_mid_noise'],
               parameters['fe4383_out'], parameters['fe4383_out_noise'], parameters['fe4383_rmax'],
               parameters['fe5270_cen'], parameters['fe5270_cen_noise'],
               parameters['fe5270_mid'], parameters['fe5270_mid_noise'],
               parameters['fe5270_out'],parameters['fe5270_out_noise'],parameters['fe5270_rmax'],
               parameters['fe5335_cen'], parameters['fe5335_cen_noise'],
               parameters['fe5335_mid'], parameters['fe5335_mid_noise'],
               parameters['fe5335_out'], parameters['fe5335_out_noise'], parameters['fe5335_rmax'],
               parameters['mgb_cen'], parameters['mgb_cen_noise'],
               parameters['mgb_mid'], parameters['mgb_mid_noise'],
               parameters['mgb_out'], parameters['mgb_out_noise'], parameters['mgb_rmax'],
               parameters['n1_5199_cen'], parameters['n1_5199_mid'], parameters['n1_5199_out'],
               parameters['ha_cen'], parameters['ha_mid'], parameters['ha_out'],
               parameters['o3_4960_cen'], parameters['o3_4960_mid'], parameters['o3_4960_out'],
               parameters['o3_5008_cen'], parameters['o3_5008_mid'], parameters['o3_5008_out'],
               parameters['dn4000_cen'], parameters['dn4000_cen_noise'],
               parameters['dn4000_mid'], parameters['dn4000_mid_noise'],
               parameters['dn4000_out'], parameters['dn4000_out_noise'], parameters['dn4000_rmax'],
               parameters['mgfe_cen'], parameters['mgfe_cen_noise'],
               parameters['mgfe_mid'], parameters['mgfe_mid_noise'],
               parameters['mgfe_out'], parameters['mgfe_out_noise'],
               parameters['mgbfe_cen'], parameters['mgbfe_cen_noise'],
               parameters['mgbfe_mid'], parameters['mgbfe_mid_noise'],
               parameters['mgbfe_out'], parameters['mgbfe_out_noise'],
               parameters['sigma_1kpc'], parameters['sigma_re_8']],
              names=['plateifu', 'ttype', 'gzoo',
                     'lummax', 'lumout(max-30kpc)', 'lum20kpc',
                     'lumcen', 'lumout_re', 'lumout(max-20kpc)', 'lum10kpc',
                     'massmax', 'massout(max-30kpc)', 'mass20kpc',
                     'masscen', 'massout_re', 'massout(max-20kpc)', 'mass10kpc',
                     'rmanga',
                     'snr_cen', 'snr_mid', 'snr_out',
                     'all_cen', 'all_mid', 'all_out',
                     'taken_cen', 'taken_mid', 'taken_out',
                     'sigma_cen', 'sigma_cen_noise',
                     'sigma_mid', 'sigma_mid_noise',
                     'sigma_out', 'sigma_out_noise', 'sigma_rmax',
                     'cn1_cen', 'cn1_cen_noise',
                     'cn1_mid', 'cn1_mid_noise',
                     'cn1_out', 'cn1_out_noise', 'cn1_rmax',
                     'ca4227_cen', 'ca4227_cen_noise',
                     'ca4227_mid', 'ca4227_mid_noise',
                     'ca4227_out', 'ca4227_out_noise', 'ca4227_rmax',
                     'fe4383_cen', 'fe4383_cen_noise',
                     'fe4383_mid', 'fe4383_mid_noise',
                     'fe4383_out', 'fe4383_out_noise', 'fe4383_rmax',
                     'fe5270_cen', 'fe5270_cen_noise',
                     'fe5270_mid', 'fe5270_mid_noise',
                     'fe5270_out', 'fe5270_out_noise', 'fe5270_rmax',
                     'fe5335_cen', 'fe5335_cen_noise',
                     'fe5335_mid', 'fe5335_mid_noise',
                     'fe5335_out', 'fe5335_out_noise', 'fe5335_rmax',
                     'mgb_cen', 'mgb_cen_noise',
                     'mgb_mid', 'mgb_mid_noise',
                     'mgb_out', 'mgb_out_noise', 'mgb_rmax',
                     'n1_5199_cen', 'n1_5199_mid', 'n1_5199_out',
                     'ha_cen', 'ha_mid', 'ha_out',
                     'o3_4960_cen', 'o3_4960_mid', 'o3_4960_out',
                     'o3_5008_cen', 'o3_5008_mid', 'o3_5008_out',
                     'dn4000_cen', 'dn4000_cen_noise',
                     'dn4000_mid', 'dn4000_mid_noise',
                     'dn4000_out', 'dn4000_out_noise', 'dn4000_rmax',
                     'mgfe_cen', 'mgfe_cen_noise',
                     'mgfe_mid', 'mgfe_mid_noise',
                     'mgfe_out', 'mgfe_out_noise',
                     'mgbfe_cen', 'mgbfe_cen_noise',
                     'mgbfe_mid', 'mgbfe_mid_noise',
                     'mgbfe_out', 'mgbfe_out_noise',
                     'sigma_1kpc', 'sigma_re_8'])
    t.write('parameters_spx2_max_' + cens[cen_num] + '.fits', overwrite=True)
    '''
    # np.save('rmax',r_max)
    # np.save('cens',cens)
    # np.save('reffs',reffs)
    t = Table([manga_sga[1].data['plateifu'], discarded_counts['DEC_30'],
               discarded_counts['no_sga_ellp'], discarded_counts['no_maps'],
               discarded_counts['zero_maps_size'], discarded_counts['no_reff'],
               discarded_counts['no_sga_prof'], discarded_counts['cen<psf'],
               discarded_counts['small_pixnum'], discarded_counts['pipe3d'],
               discarded_counts['no_maps_url'], discarded_counts['no_sga_url'],
               mask_m, mask_o, mask_e, mask_nm, mask_psf,mask_visual,mask_spec, mask_reobs,
               rmax_manga],
              names=['plateifu', 'DEC_30', 'no_sga_ellp', 'no_maps',
                     'zero_maps_size', 'no_reff', 'no_sga_prof',
                     'cen<psf', 'small_pixnum', 'pipe3d', 'no_maps_url',
                     'no_sga_url', 'mass11.2', 'outliers_mlcr', 'ba0.3',
                     'negative_mass', 'outliers_psf','outliers_visual','outliers_spec','reobserved',
                     'rmax_manga'])
    t.write('sample_mask.fits', overwrite=True)
    '''
