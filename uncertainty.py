import numpy as np
from astropy.io import fits
import pyphot
from pyphot import unit
from astropy.table import Table
import sys

split_num = int(sys.argv[1])

degs = 15
lam1=3806 #red

idx_path = './data/index'
file_path = './data/'

parameters = fits.open('parameters_reff2_2.fits')
split = fits.open('sample_split_outliers_total2.fits')
stacked = fits.open('stacked_smooth_sigma300_mask_new2.fits')

split_type = ['m10kpc_v_m20kpc','m20kpc_v_sigmacen','mre_v_m2re']
pca_label = ['no', '1', '2', '3', '4', '5']
names = np.array(['CN_1', 'CN_2', 'Ca4227', 'Fe4383', 'Ca4455','H_beta',
                  'Fe4531', 'Fe5015', 'Mg_b', 'Fe5270', 'Fe5335','TiO3','TiO_1','TiO_2','TiO_3','TiO_4',
                  'Fe5406', 'Fe5709', 'Fe5782', 'MgFe', 'Mgb/Fe'])
labels = ['high', 'low']

l = pyphot.LickLibrary()

index_in, index_mid, index_out = {}, {}, {}
for name in names:
    index_in[name] = [[], []]
    index_mid[name] = [[], []]
    index_out[name] = [[], []]

plateifus = stacked['info'].data['plateifu']
wave = stacked['WAVE'].data[:-1]
wave = wave / (1.0 + 0.05792105 / (238.0185 - (10000.0 / wave) ** 2) + 0.00167917 / (57.362 - (10000.0 / wave) ** 2))

parameters = fits.open('parameters_reff2.fits')
gals = parameters[1].data['plateifu']
_, idx, _ = np.intersect1d(plateifus, split[1].data['plateifu'], return_indices=True)

flux_in = stacked['flux_in'].data[idx]
flux_mid = stacked['flux_mid'].data[idx]
flux_out = stacked['flux_out'].data[idx]
flux_in = flux_in[:, :-1]
flux_mid = flux_mid[:, :-1]
flux_out = flux_out[:, :-1]

fnor_in = np.zeros_like(flux_in)
fnor_mid = np.zeros_like(flux_mid)
fnor_out = np.zeros_like(flux_out)

nor_in = np.zeros_like(flux_in)
nor_mid = np.zeros_like(flux_mid)
nor_out = np.zeros_like(flux_out)

for k in range(len(idx)):
    idx_nan = np.isfinite(flux_in[k])
    p0_in = np.polyfit(wave[idx_nan], flux_in[k][idx_nan], degs)
    idx_nan = np.isfinite(flux_mid[k])
    p0_mid = np.polyfit(wave[idx_nan], flux_mid[k][idx_nan], degs)
    idx_nan = np.isfinite(flux_out[k])
    p0_out = np.polyfit(wave[idx_nan], flux_out[k][idx_nan], degs)

    p_in = np.poly1d(p0_in)
    p_mid = np.poly1d(p0_mid)
    p_out = np.poly1d(p0_out)

    nor_in[k] = p_in(wave)
    nor_mid[k] = p_mid(wave)
    nor_out[k] = p_out(wave)

    fnor_in[k] = flux_in[k] / nor_in[k]
    fnor_mid[k] = flux_mid[k] / nor_mid[k]
    fnor_out[k] = flux_out[k] / nor_out[k]

mask_in = stacked['mask_in'].data[idx]
mask_mid = stacked['mask_mid'].data[idx]
mask_out = stacked['mask_out'].data[idx]
flux_in0 = np.copy(fnor_in)
flux_mid0 = np.copy(fnor_mid)
flux_out0 = np.copy(fnor_out)
flux_in0[mask_in == 0] = np.nan
flux_mid0[mask_mid == 0] = np.nan
flux_out0[mask_out == 0] = np.nan

for i in range(2):
    spec_in, spec_mid, spec_out = [], [], []
    mask = split[1].data[split_type[split_num]] % 2 == i
    _, idx0, _ = np.intersect1d(gals, split[1].data['plateifu'][mask], return_indices=True)

    f_in = np.nanmedian(flux_in0[mask], axis=0)
    f_mid = np.nanmedian(flux_mid0[mask], axis=0)
    f_out = np.nanmedian(flux_out0[mask], axis=0)

    for k in sorted(l.content):
        fk = l[k]
        if not np.array(names).__contains__(fk.name):
            continue
        try:
            index_in[fk.name][i].append(fk.get(wave[~np.isnan(f_in)] * unit('AA'), f_in[~np.isnan(f_in)], axis=1))
        except:
            index_in[fk.name][i].append(np.nan)
        try:
            index_mid[fk.name][i].append(fk.get(wave[~np.isnan(f_mid)] * unit('AA'), f_mid[~np.isnan(f_mid)], axis=1))
        except:
            index_mid[fk.name][i].append(np.nan)
        try:
            index_out[fk.name][i].append(fk.get(wave[~np.isnan(f_out)] * unit('AA'), f_out[~np.isnan(f_out)], axis=1))
        except:
            index_out[fk.name][i].append(np.nan)
    index_in['MgFe'][i].append(
        np.sqrt(index_in['Mg_b'][i][-1] * (
                0.72 * index_in['Fe5270'][i][-1] + 0.28 * index_in['Fe5335'][i][-1])))
    index_in['Mgb/Fe'][i].append(
        index_in['Mg_b'][i][-1] / (
                0.5 * (index_in['Fe5270'][i][-1] + index_in['Fe5335'][i][-1])))

    index_mid['MgFe'][i].append(
        np.sqrt(index_mid['Mg_b'][i][-1] * (
                0.72 * index_mid['Fe5270'][i][-1] + 0.28 * index_mid['Fe5335'][i][-1])))
    index_mid['Mgb/Fe'][i].append(
        index_mid['Mg_b'][i][-1] / (0.5 * (index_mid['Fe5270'][i][-1] + index_mid['Fe5335'][i][-1])))

    index_out['MgFe'][i].append(
        np.sqrt(index_out['Mg_b'][i][-1] * (
                0.72 * index_out['Fe5270'][i][-1] + 0.28 * index_out['Fe5335'][i][-1])))
    index_out['Mgb/Fe'][i].append(
        index_out['Mg_b'][i][-1] / (0.5 * (index_out['Fe5270'][i][-1] + index_out['Fe5335'][i][-1])))

    for niter in range(5):
        print('stop')
        np.random.seed(niter)
        idx_iter = np.random.choice(idx0, size=len(idx0))

        f_in = np.nanmedian(flux_in0[idx_iter], axis=0)
        f_mid = np.nanmedian(flux_mid0[idx_iter], axis=0)
        f_out = np.nanmedian(flux_out0[idx_iter], axis=0)

        spec_in.append(np.nanmedian(flux_in0[idx_iter], axis=0))
        spec_mid.append(np.nanmedian(flux_mid0[idx_iter], axis=0))
        spec_out.append(np.nanmedian(flux_out0[idx_iter], axis=0))
        for k in sorted(l.content):
            fk = l[k]
            if not np.array(names).__contains__(fk.name):
                continue
            try:
                index_in[fk.name][i].append(fk.get(wave[~np.isnan(f_in)] * unit('AA'), f_in[~np.isnan(f_in)], axis=1))
            except:
                index_in[fk.name][i].append(np.nan)
                print('failed to converge spec_in,', niter, fk.name)

            try:
                index_mid[fk.name][i].append(
                    fk.get(wave[~np.isnan(f_mid)] * unit('AA'), f_mid[~np.isnan(f_mid)], axis=1))
            except:
                index_mid[fk.name][i].append(np.nan)
                print('failed to converge spec_in,', niter, fk.name)

            try:
                index_out[fk.name][i].append(
                    fk.get(wave[~np.isnan(f_out)] * unit('AA'), f_out[~np.isnan(f_out)], axis=1))
            except:
                index_out[fk.name][i].append(np.nan)
                print('failed to converge spec_in,', niter, fk.name)

        index_in['MgFe'][i].append(
            np.sqrt(index_in['Mg_b'][i][-1] * (
                    0.72 * index_in['Fe5270'][i][-1] + 0.28 * index_in['Fe5335'][i][-1])))
        index_in['Mgb/Fe'][i].append(
            index_in['Mg_b'][i][-1] / (
                    0.5 * (index_in['Fe5270'][i][-1] + index_in['Fe5335'][i][-1])))

        index_mid['MgFe'][i].append(
            np.sqrt(index_mid['Mg_b'][i][-1] * (
                    0.72 * index_mid['Fe5270'][i][-1] + 0.28 * index_mid['Fe5335'][i][-1])))
        index_mid['Mgb/Fe'][i].append(
            index_mid['Mg_b'][i][-1] / (
                    0.5 * (index_mid['Fe5270'][i][-1] + index_mid['Fe5335'][i][-1])))

        index_out['MgFe'][i].append(
            np.sqrt(index_out['Mg_b'][i][-1] * (
                    0.72 * index_out['Fe5270'][i][-1] + 0.28 * index_out['Fe5335'][i][-1])))
        index_out['Mgb/Fe'][i].append(
            index_out['Mg_b'][i][-1] / (
                    0.5 * (index_out['Fe5270'][i][-1] + index_out['Fe5335'][i][-1])))

    t_in = Table([index_in[j][i] for j in [*index_in]], names=names)
    t_in.write(
        idx_path + 'indices_linear_deg_in_etg_' + split_type[split_num] + '_total_' +
        labels[i] + '.fits',
        overwrite=True)
    t_mid = Table([index_mid[j][i] for j in [*index_mid]], names=names)
    t_mid.write(
        idx_path + 'indices_linear_deg_mid_etg_' + split_type[split_num] + '_total_' +
        labels[i] + '.fits',
        overwrite=True)
    t_out = Table([index_out[j][i] for j in [*index_out]], names=names)
    t_out.write(
        idx_path + 'indices_linear_deg_out_etg_' + split_type[split_num] + '_total_' +
        labels[i] + '.fits',
        overwrite=True)
    primary_hdu = fits.PrimaryHDU(np.zeros(2))
    mywave = fits.ImageHDU(stacked['WAVE'].data[:-1], name='wave')
    std_in = fits.ImageHDU(np.nanstd(np.array(spec_in), axis=0), name='std_in')
    std_mid = fits.ImageHDU(np.nanstd(np.array(spec_mid), axis=0), name='std_mid')
    std_out = fits.ImageHDU(np.nanstd(np.array(spec_out), axis=0), name='std_out')
    hdul = fits.HDUList([primary_hdu, mywave, std_in, std_mid, std_out])
    hdul.writeto(idx_path + 'std_median_red_' + split_type[split_num] + '_total_' +
                 labels[i] + '.fits', overwrite=True)
