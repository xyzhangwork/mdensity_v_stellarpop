import astropy.io.fits as fits
import numpy as np
import pyphot
import astropy.constants
import sys

l = pyphot.LickLibrary()
C = astropy.constants.c.to('km/s').value
fk_na = l['Na_D']
fk_fe = l['Fe5015']

split = fits.open('sample_split_outliers_total.fits')
split_type = ['mtot_v_m20kpc', 'm10kpc_v_sigmacen', 'm10kpc_v_m20kpc_control', 'm20kpc_v_sigmacen', 'sigma_v_mtot']
split_num = int(sys.argv[1])
stacked_path='/Volumes/ZXYDisk/new_stack/'
stacked = fits.open(stacked_path + 'stacked_smooth_sigma300_mask.fits')
#lam1=3806 #red
#lam1=2404 #blue
lam1= 4367#feh
plateifus = stacked['info'].data['plateifu']
wave = stacked['WAVE'].data[432:lam1]
wave0 = stacked['WAVE'].data
dlambda = 5000 * 300 / C
target_sigma = dlambda * C / wave0
target_sigma = target_sigma[432:lam1]
labels = ['high', 'low']
for i in range(2):
    std = fits.open(stacked_path + 'std_median_red_'+split_type[split_num]+'_total_' + labels[i] + '.fits')
    file_out = '/work/xiaoya/alf/indata/'
    mask = split[1].data[split_type[split_num]] == i
    gals, idx, _ = np.intersect1d(plateifus, split[1].data['plateifu'][mask], return_indices=True)

    size = len(idx)
    wave_size = len(wave)

    flux_in = stacked['flux_in'].data[idx]
    flux_mid = stacked['flux_mid'].data[idx]
    flux_out = stacked['flux_out'].data[idx]
    flux_in = flux_in[:, 432:lam1]
    flux_mid = flux_mid[:, 432:lam1]
    flux_out = flux_out[:, 432:lam1]

    mask_in = stacked['mask_in'].data[idx]
    mask_mid = stacked['mask_mid'].data[idx]
    mask_out = stacked['mask_out'].data[idx]
    mask_in = mask_in[:, 432:lam1]
    mask_mid = mask_mid[:, 432:lam1]
    mask_out = mask_out[:, 432:lam1]

    ivar_in = stacked['ivar_in'].data[idx]
    ivar_mid = stacked['ivar_mid'].data[idx]
    ivar_out = stacked['ivar_out'].data[idx]
    ivar_in = ivar_in[:, 432:lam1]
    ivar_mid = ivar_mid[:, 432:lam1]
    ivar_out = ivar_out[:, 432:lam1]

    fnor_in = np.zeros_like(flux_in)
    fnor_mid = np.zeros_like(flux_mid)
    fnor_out = np.zeros_like(flux_out)

    nor_in = np.zeros_like(flux_in)
    nor_mid = np.zeros_like(flux_mid)
    nor_out = np.zeros_like(flux_out)

    flux_in0 = np.copy(flux_in)
    flux_mid0 = np.copy(flux_mid)
    flux_out0 = np.copy(flux_out)

    flux_in0[mask_in == 0] = np.nan
    flux_mid0[mask_mid == 0] = np.nan
    flux_out0[mask_out == 0] = np.nan

    flux_gals = np.ma.zeros((size * 3, wave_size))
    ivar_gals = np.ma.zeros((size * 3, wave_size))
    std_gals = np.ma.zeros((size * 3, wave_size))
    mask_gals = np.ma.zeros((size * 3, wave_size))

    for k in range(len(idx)):
        mask_tmp = (~np.isnan(flux_in[k]))
        p0_in = np.polyfit(wave[mask_tmp], flux_in[k][mask_tmp], 10)
        mask_tmp = (~np.isnan(flux_mid[k]))
        p0_mid = np.polyfit(wave[mask_tmp], flux_mid[k][mask_tmp], 10)
        mask_tmp = (~np.isnan(flux_out[k]))
        p0_out = np.polyfit(wave[mask_tmp], flux_out[k][mask_tmp], 10)

        p_in = np.poly1d(p0_in)
        p_mid = np.poly1d(p0_mid)
        p_out = np.poly1d(p0_out)

        nor_in[k] = p_in(wave)
        nor_mid[k] = p_mid(wave)
        nor_out[k] = p_out(wave)

        fnor_in[k] = flux_in[k] / nor_in[k]
        fnor_mid[k] = flux_mid[k] / nor_mid[k]
        fnor_out[k] = flux_out[k] / nor_out[k]

        mask_in[k][np.isnan(fnor_in[k])] = 0
        mask_mid[k][np.isnan(fnor_mid[k])] = 0
        mask_out[k][np.isnan(fnor_out[k])] = 0

        flux_gals.data[k] = flux_in[k]
        flux_gals.data[k + size] = flux_mid[k]
        flux_gals.data[k + size * 2] = flux_out[k]

        ivar_gals.data[k] = ivar_in[k]
        ivar_gals.data[k + size] = ivar_mid[k]
        ivar_gals.data[k + size * 2] = ivar_out[k]

        mask_gals.data[k] = mask_in[k]
        mask_gals.data[k + size] = mask_mid[k]
        mask_gals.data[k + size * 2] = mask_out[k]

    fnor_in0 = np.copy(fnor_in)
    fnor_in0[mask_in == 0] = np.nan
    f_in=np.nanmedian(fnor_in0,axis=0)
    fnor_mid0 = np.copy(fnor_mid)
    fnor_mid0[mask_mid == 0] = np.nan
    f_mid=np.nanmedian(fnor_mid0,axis=0)
    fnor_out0 = np.copy(fnor_out)
    fnor_out0[mask_out == 0] = np.nan
    f_out=np.nanmedian(fnor_out0,axis=0)

    std_in = std['STD_IN'].data
    std_mid = std['STD_mid'].data
    std_out = std['STD_out'].data

    weights = np.ones_like(wave)
    weights[(wave >= fk_fe.band.magnitude[0]) & (wave <= fk_fe.band.magnitude[1])] = 0

    with open(file_out + labels[i] + '_' + split_type[split_num] + '_skp_in.dat', 'w') as f:
        f.write('# 0.40 0.497 \n' + '# 0.507 0.57\n' + '# 0.57 0.63\n')
        for ig, gal in enumerate(wave):
            f.write('     ' +
                    format(wave[ig], '.4f') + '     ' +
                    format(f_in[ig], '.4f') + '     ' +
                    format(std_in[ig], '.4f') + '     ' +
                    format(weights[ig], '.4f') + '     ' +
                    format(target_sigma[ig], '.4f') + '     ' +
                    '\n')
    with open(file_out + labels[i] + '_' + split_type[split_num] + '_skp_mid.dat', 'w') as f:
        f.write('# 0.40 0.497 \n' + '# 0.507 0.57\n' + '# 0.57 0.63\n')
        for ig, gal in enumerate(wave):
            f.write('     ' +
                    format(wave[ig], '.4f') + '     ' +
                    format(f_mid[ig], '.4f') + '     ' +
                    format(std_mid[ig], '.4f') + '     ' +
                    format(weights[ig], '.4f') + '     ' +
                    format(target_sigma[ig], '.4f') + '     ' +
                    '\n')
    with open(file_out + labels[i] + '_' + split_type[split_num] + '_skp_out.dat', 'w') as f:
        f.write('# 0.40 0.497 \n' + '# 0.507 0.57\n' + '# 0.57 0.63\n')
        for ig, gal in enumerate(wave):
            f.write('     ' +
                    format(wave[ig], '.4f') + '     ' +
                    format(f_out[ig], '.4f') + '     ' +
                    format(std_out[ig], '.4f') + '     ' +
                    format(weights[ig], '.4f') + '     ' +
                    format(target_sigma[ig], '.4f') + '     ' +
                    '\n')
