import numpy as np
from astropy.io import fits
import pyphot
from pyphot import unit
from astropy.table import Table
import sys
import empca

pcas = [None, 1, 2, 3, 4, 5]

split_num = int(sys.argv[1])

degs = 10
#lam1=3806 #red
#lam1 = 2404  # blue
lam1= 4367#feh
idx_path = '/home/xiaoya/sample_select/'
file_path = '/home/xiaoya/sample_select/'

parameters = fits.open('parameters_spx2_max_reff2.fits')
split = fits.open('sample_split_outliers_total.fits')
stacked = fits.open(file_path + 'stacked_smooth_sigma300_mask.fits')

split_type = ['mtot_v_m20kpc', 'm10kpc_v_sigmacen', 'm10kpc_v_m20kpc_control', 'm20kpc_v_sigmacen', 'sigma_v_mtot']
pca_label = ['no', '1', '2', '3', '4', '5']
names = np.array(['CN_1', 'CN_2', 'Ca4227', 'Fe4383', 'Ca4455',
                  'Fe4531', 'Fe5015', 'Mg_b', 'Fe5270', 'Fe5335','TiO3','TiO_1','TiO_2','TiO_3','TiO_4',
                  'Fe5406', 'Fe5709', 'Fe5782', 'MgFe', 'Mgb/Fe'])
labels = ['high', 'low']

l = pyphot.LickLibrary()

index_in, index_mid, index_out = {}, {}, {}
for name in names:
    index_in[name] = [[[], [], [], [], [], []], [[], [], [], [], [], []]]
    index_mid[name] = [[[], [], [], [], [], []], [[], [], [], [], [], []]]
    index_out[name] = [[[], [], [], [], [], []], [[], [], [], [], [], []]]

plateifus = stacked['info'].data['plateifu']
wave = stacked['WAVE'].data[432:lam1]
wave = wave / (1.0 + 0.05792105 / (238.0185 - (10000.0 / wave) ** 2) + 0.00167917 / (57.362 - (10000.0 / wave) ** 2))

morph_mask = (parameters[1].data['gzoo'][:, 0] == 1) | (parameters[1].data['ttype'] > 0)

for i in range(2):
    spec_in, spec_mid, spec_out = [], [], []
    mask = split[1].data[split_type[split_num]] % 2 == i
    _, idx_ltg, _ = np.intersect1d(split[1].data['plateifu'], parameters[1].data['plateifu'][morph_mask],
                                   return_indices=True)
    mask[idx_ltg] = False

    _, idx, _ = np.intersect1d(plateifus, split[1].data['plateifu'][mask], return_indices=True)

    flux_in = stacked['flux_in'].data[idx]
    flux_mid = stacked['flux_mid'].data[idx]
    flux_out = stacked['flux_out'].data[idx]
    flux_in = flux_in[:, 432:lam1]
    flux_mid = flux_mid[:, 432:lam1]
    flux_out = flux_out[:, 432:lam1]

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
    mask_in = mask_in[:, 432:lam1]
    mask_mid = mask_mid[:, 432:lam1]
    mask_out = mask_out[:, 432:lam1]

    if np.any(np.isnan(fnor_in)):
        num_nan = np.unique(np.where(np.isnan(fnor_in))[0])
        fnor_in = np.delete(fnor_in, num_nan, axis=0)
        flux_in = np.delete(flux_in, num_nan, axis=0)
        mask_in = np.delete(mask_in, num_nan, axis=0)

    if np.any(np.isnan(fnor_mid)):
        num_nan = np.unique(np.where(np.isnan(fnor_mid))[0])
        fnor_mid = np.delete(fnor_mid, num_nan, axis=0)
        flux_mid = np.delete(flux_mid, num_nan, axis=0)
        mask_mid = np.delete(mask_mid, num_nan, axis=0)

    if np.any(np.isnan(fnor_out)):
        num_nan = np.unique(np.where(np.isnan(fnor_out))[0])
        fnor_out = np.delete(fnor_out, num_nan, axis=0)
        flux_out = np.delete(flux_out, num_nan, axis=0)
        mask_out = np.delete(mask_out, num_nan, axis=0)

    m_in = empca.empca(fnor_in, weights=mask_in, silent=True)
    m_mid = empca.empca(fnor_mid, weights=mask_mid, silent=True)
    m_out = empca.empca(fnor_out, weights=mask_out, silent=True)

    for pca in pcas:
        if pca:
            f_in = np.dot(np.nanmean(m_in.coeff[:, :pca], axis=0), m_in.eigvec[:pca, :])
            f_mid = np.dot(np.nanmean(m_mid.coeff[:, :pca], axis=0), m_mid.eigvec[:pca, :])
            f_out = np.dot(np.nanmean(m_out.coeff[:, :pca], axis=0), m_out.eigvec[:pca, :])

        else:
            pca = 0
            flux_in0 = np.copy(fnor_in)
            flux_mid0 = np.copy(fnor_mid)
            flux_out0 = np.copy(fnor_out)
            flux_in0[mask_in == 0] = np.nan
            flux_mid0[mask_mid == 0] = np.nan
            flux_out0[mask_out == 0] = np.nan

            f_in = np.nanmedian(flux_in0, axis=0)
            f_mid = np.nanmedian(flux_mid0, axis=0)
            f_out = np.nanmedian(flux_out0, axis=0)
        for k in sorted(l.content):
            fk = l[k]
            if not np.array(names).__contains__(fk.name):
                continue
            try:
                index_in[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_in, axis=1))
            except:
                index_in[fk.name][i][pca].append(np.nan)
            try:
                index_mid[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_mid, axis=1))
            except:
                index_mid[fk.name][i][pca].append(np.nan)
            try:
                index_out[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_out, axis=1))
            except:
                index_out[fk.name][i][pca].append(np.nan)

        index_in['MgFe'][i][pca].append(
            np.sqrt(index_in['Mg_b'][i][pca][-1] * (
                    0.72 * index_in['Fe5270'][i][pca][-1] + 0.28 * index_in['Fe5335'][i][pca][-1])))
        index_in['Mgb/Fe'][i][pca].append(
            index_in['Mg_b'][i][pca][-1] / (
                    0.5 * (index_in['Fe5270'][i][pca][-1] + index_in['Fe5335'][i][pca][-1])))

        index_mid['MgFe'][i][pca].append(
            np.sqrt(index_mid['Mg_b'][i][pca][-1] * (
                    0.72 * index_mid['Fe5270'][i][pca][-1] + 0.28 * index_mid['Fe5335'][i][pca][-1])))
        index_mid['Mgb/Fe'][i][pca].append(
            index_mid['Mg_b'][i][pca][-1] / (0.5 * (index_mid['Fe5270'][i][pca][-1] + index_mid['Fe5335'][i][pca][-1])))

        index_out['MgFe'][i][pca].append(
            np.sqrt(index_out['Mg_b'][i][pca][-1] * (
                    0.72 * index_out['Fe5270'][i][pca][-1] + 0.28 * index_out['Fe5335'][i][pca][-1])))
        index_out['Mgb/Fe'][i][pca].append(
            index_out['Mg_b'][i][pca][-1] / (0.5 * (index_out['Fe5270'][i][pca][-1] + index_out['Fe5335'][i][pca][-1])))

    for niter in range(1000):
        np.random.seed(niter)
        idx_iter = np.random.choice(idx, size=len(idx))
        flux_in = stacked['flux_in'].data[idx_iter]
        flux_mid = stacked['flux_mid'].data[idx_iter]
        flux_out = stacked['flux_out'].data[idx_iter]
        flux_in = flux_in[:, 432:lam1]
        flux_mid = flux_mid[:, 432:lam1]
        flux_out = flux_out[:, 432:lam1]
        mask_in = stacked['mask_in'].data[idx_iter]
        mask_mid = stacked['mask_mid'].data[idx_iter]
        mask_out = stacked['mask_out'].data[idx_iter]
        mask_in = mask_in[:, 432:lam1]
        mask_mid = mask_mid[:, 432:lam1]
        mask_out = mask_out[:, 432:lam1]

        fnor_in = np.zeros_like(flux_in)
        fnor_mid = np.zeros_like(flux_mid)
        fnor_out = np.zeros_like(flux_out)

        for k in range(len(idx_iter)):
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
        if np.any(np.isnan(fnor_in)):
            num_nan = np.unique(np.where(np.isnan(fnor_in))[0])
            fnor_in = np.delete(fnor_in, num_nan, axis=0)
            flux_in = np.delete(flux_in, num_nan, axis=0)
            mask_in = np.delete(mask_in, num_nan, axis=0)

        if np.any(np.isnan(fnor_mid)):
            num_nan = np.unique(np.where(np.isnan(fnor_mid))[0])
            fnor_mid = np.delete(fnor_mid, num_nan, axis=0)
            flux_mid = np.delete(flux_mid, num_nan, axis=0)
            mask_mid = np.delete(mask_mid, num_nan, axis=0)

        if np.any(np.isnan(fnor_out)):
            num_nan = np.unique(np.where(np.isnan(fnor_out))[0])
            fnor_out = np.delete(fnor_out, num_nan, axis=0)
            flux_out = np.delete(flux_out, num_nan, axis=0)
            mask_out = np.delete(mask_out, num_nan, axis=0)

        try:
            m_in = empca.empca(fnor_in, weights=mask_in, silent=True)
            m_mid = empca.empca(fnor_mid, weights=mask_mid, silent=True)
            m_out = empca.empca(fnor_out, weights=mask_out, silent=True)

        except:
            print('failed to perform empca,', niter)
            continue
        for pca in pcas:
            if pca:
                f_in = np.dot(np.nanmean(m_in.coeff[:, :pca], axis=0), m_in.eigvec[:pca, :])
                f_mid = np.dot(np.nanmean(m_mid.coeff[:, :pca], axis=0), m_mid.eigvec[:pca, :])
                f_out = np.dot(np.nanmean(m_out.coeff[:, :pca], axis=0), m_out.eigvec[:pca, :])
            else:
                pca = 0
                flux_in0 = np.copy(fnor_in)
                flux_mid0 = np.copy(fnor_mid)
                flux_out0 = np.copy(fnor_out)
                flux_in0[mask_in == 0] = np.nan
                flux_mid0[mask_mid == 0] = np.nan
                flux_out0[mask_out == 0] = np.nan

                f_in = np.nanmedian(flux_in0, axis=0)
                f_mid = np.nanmedian(flux_mid0, axis=0)
                f_out = np.nanmedian(flux_out0, axis=0)

                spec_in.append(np.nanmedian(flux_in0, axis=0))
                spec_mid.append(np.nanmedian(flux_mid0, axis=0))
                spec_out.append(np.nanmedian(flux_out0, axis=0))

            for k in sorted(l.content):
                fk = l[k]
                if not np.array(names).__contains__(fk.name):
                    continue
                try:
                    index_in[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_in, axis=1))
                except:
                    index_in[fk.name][i][pca].append(np.nan)
                    print('failed to converge spec_in,', niter, fk.name)
                try:
                    index_mid[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_mid, axis=1))
                except:
                    index_mid[fk.name][i][pca].append(np.nan)
                    print('failed to converge spec_mid,', niter, fk.name)

                try:
                    index_out[fk.name][i][pca].append(fk.get(wave * unit('AA'), f_out, axis=1))
                except:
                    index_out[fk.name][i][pca].append(np.nan)
                    print('failed to converge spec_out,', niter, fk.name)

            index_in['MgFe'][i][pca].append(
                np.sqrt(index_in['Mg_b'][i][pca][-1] * (
                        0.72 * index_in['Fe5270'][i][pca][-1] + 0.28 * index_in['Fe5335'][i][pca][-1])))
            index_in['Mgb/Fe'][i][pca].append(
                index_in['Mg_b'][i][pca][-1] / (
                        0.5 * (index_in['Fe5270'][i][pca][-1] + index_in['Fe5335'][i][pca][-1])))

            index_mid['MgFe'][i][pca].append(
                np.sqrt(index_mid['Mg_b'][i][pca][-1] * (
                        0.72 * index_mid['Fe5270'][i][pca][-1] + 0.28 * index_mid['Fe5335'][i][pca][-1])))
            index_mid['Mgb/Fe'][i][pca].append(
                index_mid['Mg_b'][i][pca][-1] / (
                        0.5 * (index_mid['Fe5270'][i][pca][-1] + index_mid['Fe5335'][i][pca][-1])))

            index_out['MgFe'][i][pca].append(
                np.sqrt(index_out['Mg_b'][i][pca][-1] * (
                        0.72 * index_out['Fe5270'][i][pca][-1] + 0.28 * index_out['Fe5335'][i][pca][-1])))
            index_out['Mgb/Fe'][i][pca].append(
                index_out['Mg_b'][i][pca][-1] / (
                        0.5 * (index_out['Fe5270'][i][pca][-1] + index_out['Fe5335'][i][pca][-1])))

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
    mywave = fits.ImageHDU(stacked['WAVE'].data[432:lam1], name='wave')
    std_in = fits.ImageHDU(np.nanstd(np.array(spec_in), axis=0), name='std_in')
    std_mid = fits.ImageHDU(np.nanstd(np.array(spec_mid), axis=0), name='std_mid')
    std_out = fits.ImageHDU(np.nanstd(np.array(spec_out), axis=0), name='std_out')
    hdul = fits.HDUList([primary_hdu, mywave, std_in, std_mid, std_out])
    hdul.writeto('std_median_feh_' + split_type[split_num] + '_total_' +
                 labels[i] + '.fits', overwrite=True)
