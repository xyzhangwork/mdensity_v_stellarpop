import numpy as np
import astropy.io.fits as fits
import sys
file_path = './data/'

manga_sga = fits.open('./data/manga_sga_z.fits')
blue_sky = np.array([5578.9, 5891.59, 5897.56, 6302.05, 6365.54, 6865.86])
red_sky = np.array([8016.27, 8027.89, 8028.17, 8054.24, 8064.21, 8064.61, 8280.59,
                     8283.97, 8288.69, 8290.89, 8298.53, 8301.19, 8313., 8345.33,
                     8346.9, 8355.18, 8384.7, 8401.49, 8417.55, 8432.48, 8454.59,
                     8467.56, 8467.84, 8495.73, 8506.98, 8507.41, 8541.04, 8550.74,
                     8630.29, 8633.87, 8637.64, 8652.1, 8653.67, 8656.74, 8658.24,
                     8661.52, 8663.02, 8666.51, 8667.97, 8671.66, 8673.11, 8678.45,
                     8761.14, 8762.69, 8763.75, 8764.03, 8768.83, 8770.33, 8778.63,
                     8780.75, 8793.6, 8827.88, 8829.53, 8838.88, 8852.03, 8852.5,
                     8870.04, 8888.3, 8905.56, 8922.09, 8945.87, 8960.41, 8960.71,
                     8990.85])
sky=[blue_sky,red_sky]


masked, unmasked = 0, 0
missing = []
stacked = fits.open('./data/stacked_sigfix.fits')
plateifus = stacked['info'].data['plateifu']
_, _, idx0 = np.intersect1d(plateifus, stacked['info'].data['plateifu'], return_indices=True)
_, _, idx = np.intersect1d(plateifus, manga_sga[1].data['plateifu'], return_indices=True)
f_in, f_mid, f_out = stacked['flux_in'].data, stacked['flux_mid'].data, stacked['flux_out'].data
mask_in, mask_mid, mask_out = np.zeros_like(stacked['flux_in'].data), np.zeros_like(
    stacked['flux_mid'].data), np.zeros_like(stacked['flux_out'].data)
resid_in, resid_mid, resid_out = stacked['resid_in'].data, stacked['resid_mid'].data, stacked['resid_out'].data
wave = stacked['wave'].data
blue_mask = (wave > 4000) & (wave < 7000)
red_mask = (wave > 8000) & (wave < 8700)
mask = [blue_mask, red_mask]

for i, gal in enumerate(plateifus):
    cat = stacked['info'].data['plateifu'][i]
    z = manga_sga[1].data['redshift'][idx[i]]
    in_tmp = np.ones_like(f_in[i])
    mid_tmp = np.ones_like(f_mid[i])
    out_tmp = np.ones_like(f_out[i])
    for j in range(2):
        median_in = np.nanmedian(resid_in[i][mask[j]])
        median_mid = np.nanmedian(resid_mid[i][mask[j]])
        median_out = np.nanmedian(resid_out[i][mask[j]])

        std_in = np.nanpercentile(resid_in[i][mask[j]], 84) - median_in
        std_mid = np.nanpercentile(resid_mid[i][mask[j]], 84) - median_mid
        std_out = np.nanpercentile(resid_out[i][mask[j]], 84) - median_out
        mask0_in = ~((resid_in[i] < median_in + std_in) & (resid_in[i] > median_in - std_in))
        mask0_mid = ~((resid_in[i] < median_mid + std_mid) & (resid_mid[i] > median_mid - std_mid))
        mask0_out = ~((resid_out[i] < median_out + std_out) & (resid_out[i] > median_out - std_out))
        for k in range(len(sky[j])):
            in_tmp[mask[j] & mask0_in & (abs(wave - sky[j][k] / (1 + z)) <= 3)] = 0
            mid_tmp[mask[j] & mask0_mid & (abs(wave - sky[j][k] / (1 + z)) <= 3)] = 0
            out_tmp[mask[j] & mask0_out & (abs(wave - sky[j][k] / (1 + z)) <= 3)] = 0
        in_tmp[(~((resid_in[i] < median_in + 5 * std_in) & (
                resid_in[i] > median_in - 5 * std_in))) & mask[j]] = 0
        mid_tmp[mask[j] & (~((resid_mid[i] < median_mid + 5 * std_mid) & (
                resid_mid[i] > median_mid - 5 * std_mid)))] = 0
        out_tmp[mask[j] & (~((resid_out[i] < median_out + 5 * std_out) & (
                resid_out[i] > median_out - 5 * std_out)))] = 0
    stacked['mask_in'].data[idx0[i]] = np.array(in_tmp)
    stacked['mask_mid'].data[idx0[i]] = np.array(mid_tmp)
    stacked['mask_out'].data[idx0[i]] = np.array(out_tmp)

stacked.writeto('stacked_sigfix_mask.fits', overwrite=True)
