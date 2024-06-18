from astropy.io import fits
import astropy.cosmology as cosmology
from calc_kcor import calc_kcor
import matplotlib.pyplot as plt
import os
import mclrs
from scipy import interpolate
import numpy as np
from plot_cogl import get_cog
from astropy.table import Table

if __name__ == '__main__':
    cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)
    mask_hdu = fits.open('/home/xiaoya/sample_select/sample_mask.fits')

    path_sga = '/home/xiaoya/sample_select/tmp/'
    manga_sga = fits.open('/home/xiaoya/sample_select/manga_sga_z.fits')
    p = fits.open('/home/xiaoya/sample_select/SDSS17Pipe3D_v3_1_1.fits')

    sga = fits.open('/home/xiaoya/sample_select/SGA-2020.fits')
    mask_coord = manga_sga[1].data['objdec'] < 30
    mask_mass = manga_sga[1].data['m'] > 11.2
    # filter information: https://www.legacysurvey.org/dr9/description/#photometry

    factor_decalz = {'g': [0.08804478, -0.01202266],
                     'r': [0.097115220, -0.005264246],
                     'z': [0.0504410920, -0.0002501634]}
    factor_bass = {'g': [0.011121011, -0.009446287],
                   'r': [0.080892829, -0.003455136],
                   'z': [0.073761688, 0.004807630]}

    extinctionCoeffs = {
        "u": 3.994,
        "g": 3.212,
        "r": 2.164,
        "i": 1.591,
        "z": 1.211,
        "y": 1.063,
        "N540": 2.753,
        "N708": 1.847,
        }
    r_band = {'g': 3.214, 'r': 2.165, 'i': 1.592, 'z': 1.211}
    # https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients

    solar_l = {'u': 6.39, 'g': 5.11, 'r': 4.65, 'i': 4.53,
               'z': 4.5}  # SDSS filters, from http://mips.as.arizona.edu/~cnaw/sun.html
    filter_cor = {'g': 0.0282, 'r': 0.0381,
                  'z': 0.0092}  # filter tranformation https://transformcalc.icrar.org/
    fig, ax = plt.subplots(clear=True)
    chosen = manga_sga[1].data[mask_coord]['plateifu']
    _, idx, idx_manga = np.intersect1d(chosen, manga_sga[1].data['plateifu'], return_indices=True)
    mass_sga0, mass_sga1, ifus, gamma, re_kpc, re_arc, re_nsa_arc, re_nsa_kpc = [], [], [], [], [], [], [], []
    mass_sga0_2, mass_sga1_2=[],[]
    rmax_arc = []
    psf_g, psf_r, psf_z = [], [], []
    dx_cog, dy_cog, ifu_cog = [], [], []
    gr_colors1,gr_colors2=[],[]
    samp_color = [[], []]
    re_cog = []
    counting = 0
    print('files_opened_right_before_loop', os.system('lsof -u xiaoya | wc -l'))
    ba = []
    pa=[]
    mask_names = ['no_maps',
                  'zero_maps_size',
                  'no_maps_url',
                  'no_sga_url',
                  'no_sga_ellp',
                  'no_reff',
                  'no_sga_prof']
    mask = mask_hdu[1].data[mask_names[0]] != 2
    for i in range(len(mask_names)):
        mask = mask & (mask_hdu[1].data[mask_names[i]] == 0)
    for gal in manga_sga[1].data:
        ifu = gal['plateifu']
        idx_msk = np.where(mask_hdu[1].data['plateifu'] == ifu)[0][0]
        if not mask[idx_msk]:
            continue
        sid = gal['sgaid']
        idx_sga = np.where(sga[1].data['sga_id'] == sid)[0][0]
        idx_pipe3d = np.where(p[1].data['plateifu'] == ifu)[0]
        gn = sga[1].data['group_name'][idx_sga]
        ra = gal['objra']
        dec = gal['objdec']

        sid = str(sid)
        file_path = 'sga ellipse file path'
        maps_path = 'manga maps file path'

        hdu = fits.open(file_path + gn + '-largegalaxy-' + sid + '-ellipse.fits')
        hdap = fits.open(maps_path + 'manga-' + ifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz')
        position_angle = np.copy(hdu[1].data['PA'])[0]
        z_sga = sga[1].data['z_leda'][idx_sga]
        d_l = cosm.luminosity_distance(z_sga).to('pc').value
        d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value

        ebv = hdap[0].header['EBVGAL']
        reff = hdap[0].header['REFF']
        # transform unit from nanomaggies/arcsecond^2 to magnitude/arcsecond^2 (absolute mag)
        sga_grz_original = {
            'g': 22.5 - 2.5 * np.log10(hdu[1].data['G_INTENS'][0]) - r_band['g'] * ebv - 5 * np.log10(
                d_l / 10),
            'r': 22.5 - 2.5 * np.log10(hdu[1].data['R_INTENS'][0]) - r_band['r'] * ebv - 5 * np.log10(
                d_l / 10),
            'z': 22.5 - 2.5 * np.log10(hdu[1].data['Z_INTENS'][0]) - r_band['z'] * ebv - 5 * np.log10(
                d_l / 10)}
        snr_mask = {'g': hdu[1].data['G_INTENS'][0] / hdu[1].data['G_INTENS_ERR'][0] > 1,
                    'r': hdu[1].data['R_INTENS'][0] / hdu[1].data['R_INTENS_ERR'][0] > 1,
                    'z': hdu[1].data['Z_INTENS'][0] / hdu[1].data['Z_INTENS_ERR'][0] > 1}
        dx_color = hdu[1].data['R_SMA'][0] * 0.263
        dx = hdu[1].data['R_SMA'][0] * 0.263  # SMA in pixel, 0.263'' per pixel
        eps_r = np.copy(hdu[1].data['R_EPS'][0])[1]

        # filter tranform to sdss ugriz
        if dec < 32.375:
            sga_grz1 = {
                'g': sga_grz_original['g'] + factor_decalz['g'][0] * (sga_grz_original['g'] - sga_grz_original['r']) + factor_decalz['g'][1],
                'r': sga_grz_original['r'] + factor_decalz['r'][0] * (sga_grz_original['g'] - sga_grz_original['r']) + factor_decalz['r'][1],
                'z': sga_grz_original['z'] + factor_decalz['z'][0] * (sga_grz_original['r'] - sga_grz_original['z']) + factor_decalz['z'][1]}
        else:
            sga_grz1 = {
                'g': sga_grz_original['g'] + factor_bass['g'][0] * (sga_grz_original['g'] - sga_grz_original['r']) + factor_bass['g'][1],
                'r': sga_grz_original['r'] + factor_bass['r'][0] * (sga_grz_original['g'] - sga_grz_original['r']) + factor_bass['r'][1],
                'z': sga_grz_original['z'] + factor_bass['z'][0] * (sga_grz_original['r'] - sga_grz_original['z']) + factor_bass['z'][1]}

        # k-correction: http://kcor.sai.msu.ru/
        sga_grz = {
            'g': sga_grz1['g'] - calc_kcor('g', z_sga, 'gr', sga_grz1['g'] - sga_grz1['r']),
            'r': sga_grz1['r'] - calc_kcor('r', z_sga, 'gr', sga_grz1['g'] - sga_grz1['r']),
            'z': sga_grz1['z'] - calc_kcor('z', z_sga, 'gz', sga_grz1['g'] - sga_grz1['z'])
            }

        g = np.copy(sga_grz['g'][snr_mask['r']])
        r = np.copy(sga_grz['r'][snr_mask['r']])
        z = np.copy(sga_grz['z'][snr_mask['r']])
        dx_color = dx_color[snr_mask['r']]

        # unit transform into flux/arcsecond^2
        lum_r = 10 ** (- 0.4 * (sga_grz['r'] - solar_l['r']))
        lum_g = 10 ** (- 0.4 * (sga_grz['g'] - solar_l['g']))

        dlum_r = get_cog(dx / 0.263, lum_r,
                         axis_ratio=1 - eps_r) * 0.263 ** 2
        dlum_g = get_cog(dx / 0.263, lum_g,
                         axis_ratio=1 - eps_r) * 0.263 ** 2
        dlum_r = np.array(np.log10(dlum_r), dtype=float)
        dlum_g = np.array(np.log10(dlum_g), dtype=float)
        r_in = min(hdu[1].data['psfsize_g'][0], hdu[1].data['psfsize_r'][0])

        dlum_r = dlum_r[snr_mask['r']&snr_mask['g']]
        dlum_g = dlum_g[snr_mask['g']&snr_mask['r']]

        dx = dx[snr_mask['r']&snr_mask['g']]

        diff0 = np.insert(abs(np.diff(dlum_r)), 0, 0)
        diff1 = np.insert(abs(np.diff(dlum_g)), 0, 0)

        idx_diff0 = np.where((diff0 > np.nanstd(dlum_r[(dx > reff) & (dx < 1.5 * reff)])) & (dx > 1.5 * reff))[0]
        idx_diff1 = np.where((diff1 > np.nanstd(dlum_g[(dx > reff) & (dx < 1.5 * reff)])) & (dx > 1.5 * reff))[0]

        if len(idx_diff0) >= 1:
            dlum_r[idx_diff0[0]:] = np.nan
            dlum_g[idx_diff0[0]:] = np.nan
        if len(idx_diff1) >= 1:
            dlum_r[idx_diff1[0]:] = np.nan
            dlum_g[idx_diff1[0]:] = np.nan

        idx_tmp = np.where(dlum_r == np.nanmax(dlum_r))[0][0]
        lumtot = dlum_r[idx_tmp]

        r_tmp = dlum_r[:idx_tmp + 1]
        g_tmp = dlum_g[:idx_tmp + 1]

        dx_tmp = dx[:idx_tmp + 1]

        mask_tmp = (~np.isnan(r_tmp)) & (~np.isnan(g_tmp)) & (~np.isnan(dx_tmp))
        g_tmp = g_tmp[mask_tmp]
        r_tmp = r_tmp[mask_tmp]
        dx_tmp = dx_tmp[mask_tmp]
        if r_in>dx_tmp[-1]:
            continue
        func_re = interpolate.interp1d(r_tmp, dx_tmp)
        func_g = interpolate.interp1d(dx_tmp ** 0.25, g_tmp)
        func_r = interpolate.interp1d(dx_tmp ** 0.25, r_tmp)
        gr_color = -2.5 * np.log10((10 ** func_g(dx_tmp[-1] ** 0.25) - 10 ** func_g(r_in ** 0.25)) / (
                    10 ** func_r(dx_tmp[-1] ** 0.25) - 10 ** func_r(r_in ** 0.25))) + (solar_l['g'] - solar_l['r'])
        if dx_tmp[-1]>(r_in*2):
            gr_color2 = -2.5 * np.log10((10 ** func_g(dx_tmp[-1] ** 0.25) - 10 ** func_g((2*r_in) ** 0.25)) / (
                    10 ** func_r(dx_tmp[-1] ** 0.25) - 10 ** func_r((2*r_in) ** 0.25))) + (solar_l['g'] - solar_l['r'])
        else:
            gr_color2=np.nan

        lgml0 = mclrs.bc03_rc15('r', 'gr', gr_color)
        lgml0_2 = mclrs.bc03_rc15('r', 'gr', gr_color2)

        mass0 = lgml0 + lumtot
        mass0_2 = lgml0_2 + lumtot

        lgml1 = mclrs.b19_etg_r(gr_color)
        lgml1_2 = mclrs.b19_etg_r(gr_color2)

        mass1 = lgml1 + lumtot
        mass1_2 = lgml1_2 + lumtot

        if mass0<0 or mass1<0:
            continue
        psf_g.append(np.copy(hdu[1].data['psfsize_g'])[0])
        psf_r.append(np.copy(hdu[1].data['psfsize_r'])[0])
        psf_z.append(np.copy(hdu[1].data['psfsize_z'])[0])
        ba.append(1 - eps_r)
        pa.append(position_angle)
        gr_colors1.append(gr_color)
        gr_colors2.append(gr_color2)

        re_tmp = func_re(lumtot - np.log10(2))
        rmax_tmp=func_re(lumtot)
        re_arc.append(re_tmp)
        re_kpc.append(re_tmp * (d_a * np.pi / (3600 * 180)))
        re_nsa_arc.append(reff)
        re_nsa_kpc.append(reff * (d_a * np.pi / (3600 * 180)))
        rmax_arc.append(rmax_tmp)
        gamma.append(lgml0)

        mass_sga0.append(mass0)
        mass_sga0_2.append(mass0_2)

        mass_sga1_2.append(mass1_2)
        ifus.append(ifu)

    t = Table([ifus, mass_sga0,mass_sga0_2, mass_sga1,mass_sga1_2, gamma, re_arc,rmax_arc, re_kpc, re_nsa_arc, re_nsa_kpc, psf_g, psf_r, psf_z, ba,pa,gr_colors1,gr_colors2],
              names=['plateifu', 'mass_sga_rc15', 'mass2_sga_rc15','mass_sga_b19','mass2_sga_b19', 'lg_M/L', 'Re_arc','Rmax_arc', 'Re_kpc', 'Re(NSA)_arc',
                     'Re(NSA)_kpc', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'ba','pa','gr_1psf','gr_2psf'])
    t.write('sga_mass_new.fits', overwrite=True)
