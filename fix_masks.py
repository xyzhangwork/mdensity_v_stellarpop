import numpy as np
import astropy.io.fits as fits

file_path = '/Volumes/ZXYDisk/new_stack/'

# file_path = '/Users/evrinezhang/Desktop/'
stacked = fits.open(file_path + 'stacked_sigfix.fits')
# smoothed_spec =fits.open(file_path+'stacked_smooth_sigma300.fits')
manga_sga = fits.open('manga_sga_z.fits')
plateifus = stacked['info'].data['plateifu']
_, idx0, idx = np.intersect1d(plateifus, manga_sga[1].data['plateifu'], return_indices=True)
sky_list = np.array([5578.9, 5891.59, 5897.56, 6302.05, 6365.54, 6865.86])
'''
sky_list = np.array([4021.37, 4030.2, 4045.17, 4062.98, 4068.01, 4085.24, 4087.1,
                     4089.74, 4090.8, 4092.17, 4101.79, 4102.15, 4102.88, 4157.71,
                     4168.46, 4172.87, 4177.73, 4177.95, 4185.47, 4406.07, 4586.99,
                     4682.5, 5199.38, 5201.74, 5204.43, 5225.6, 5240.21, 5257.51,
                     5257.59, 5331.9, 5543.78, 5546.13, 5565.54, 5578.9, 5590.69,
                     5599.34, 5606.71, 5625.34, 5625.46, 5867.1, 5867.63, 5869.18,
                     5869.94, 5874.37, 5888.77, 5889.82, 5891.59, 5896.1, 5897.56,
                     5908.35, 5916.95, 5926.37, 5934.5, 5947.59, 5955.01, 5955.14,
                     5971.93, 5978.62, 5978.87, 5999.43, 6005.41, 6005.75, 6141.17,
                     6142.25, 6146.18, 6172.36, 6178., 6186.02, 6194.65, 6204.45,
                     6214.79, 6223.5, 6236.02, 6237.4, 6237.7, 6238.72, 6239.15,
                     6241.97, 6243.43, 6245.3, 6245.46, 6258.68, 6259.7, 6263.48,
                     6266.95, 6279.84, 6289.18, 6299.65, 6302.05, 6303.02, 6308.67,
                     6323.16, 6331.53, 6331.68, 6350.3, 6357.94, 6358.21, 6365.54,
                     6381.08, 6388.02, 6388.39, 6415.54, 6415.66, 6421.85, 6422.33,
                     6465.32, 6465.59, 6467.74, 6471.94, 6472.77, 6479.71, 6499.33,
                     6500.54, 6506.8, 6515.65, 6515.95, 6524.23, 6534.85, 6545.85,
                     6555.44, 6564.57, 6570.6, 6579.02, 6579.2, 6598.46, 6605.82,
                     6606.1, 6629.46, 6635.88, 6636.26, 6663.53, 6712.99, 6718.82,
                     6829.35, 6830.36, 6831.39, 6831.81, 6834.56, 6835.91, 6836.34,
                     6841.96, 6843.85, 6864.59, 6865.86, 6872.97, 6882.85, 6883.15,
                     6891.2, 6902.71, 6914.55, 6925.1, 6941.45, 6950.88, 6951.09,
                     6953.62, 6956.48, 6968.26, 6971.86, 6980.19, 6980.5])
'''
f_in, f_mid, f_out = stacked['flux_in'].data, stacked['flux_mid'].data, stacked['flux_out'].data
mask_in, mask_mid, mask_out = np.zeros_like(stacked['flux_in'].data), np.zeros_like(
    stacked['flux_mid'].data), np.zeros_like(stacked['flux_out'].data)
resid_in, resid_mid, resid_out = stacked['resid_in'].data, stacked['resid_mid'].data, stacked['resid_out'].data
# mask0_in, mask0_mid, mask0_out = stacked['mask_in'].data, stacked['mask_mid'].data, stacked['mask_out'].data

wave = stacked['wave'].data
wave_mask = (wave > 4000) & (wave < 7000)
masked, unmasked = 0, 0
missing = []
for i in range(len(f_in)):
    cat = stacked['info'].data['plateifu'][i]
    # if stacked['info'].data['plateifu'][i] != '8997-12704':
    #    continue
    z = manga_sga[1].data['redshift'][idx[i]]
    # resid_tmp_in = abs(resid_in[i])
    std_in = np.nanstd(resid_in[i][(resid_in[i] < np.nanpercentile(resid_in[i], 90)) & (
            resid_in[i] > np.nanpercentile(resid_in[i], 10))])
    std_mid = np.nanstd(resid_mid[i][(resid_mid[i] < np.nanpercentile(resid_mid[i], 90)) & (
            resid_mid[i] > np.nanpercentile(resid_mid[i], 10))])
    std_out = np.nanstd(resid_out[i][(resid_out[i] < np.nanpercentile(resid_out[i], 90)) & (
            resid_out[i] > np.nanpercentile(resid_out[i], 10))])
    median_in = np.nanmedian(resid_in[i][wave_mask])
    median_mid = np.nanmedian(resid_mid[i][wave_mask])
    median_out = np.nanmedian(resid_out[i][wave_mask])
    # resid_tmp_mid = abs(resid_mid[i])
    # resid_tmp_out = abs(resid_out[i])
    in_tmp = np.ones_like(f_in[i])
    mid_tmp = np.ones_like(f_mid[i])
    out_tmp = np.ones_like(f_out[i])
    mask0_in = ~((resid_in[i] < median_in + 1 * std_in) & (resid_in[i] > median_in - 1 * std_in))
    mask0_mid = ~((resid_in[i] < median_mid + 1 * std_mid) & (resid_mid[i] > median_mid - 1 * std_mid))
    mask0_out = ~((resid_out[i] < median_out + 1 * std_out) & (resid_out[i] > median_out - 1 * std_out))

    if np.nansum(1 * (abs(sky_list / (1 + z) - 5287) < 5)) > 0:
        if np.nansum(1 * (mask_in[i] == 1)):
            # print(i,z,'not masked')
            unmasked += 1
        else:
            # print(i,z,'masked')
            masked += 1
    for j in range(len(sky_list)):
        in_tmp[mask0_in & (abs(wave - sky_list[j] / (1 + z)) <= 3)] = 0
        mid_tmp[mask0_mid & (abs(wave - sky_list[j] / (1 + z)) <= 3)] = 0
        out_tmp[mask0_out & (abs(wave - sky_list[j] / (1 + z)) <= 3)] = 0
    in_tmp[~((resid_in[i] < median_in + 10 * std_in) & (
            resid_in[i] > median_in - 10 * std_in))] = 0
    mid_tmp[~((resid_mid[i] < median_mid + 10 * std_mid) & (
            resid_mid[i] > median_mid - 10 * std_mid))] = 0
    out_tmp[~((resid_out[i] < median_out + 10 * std_out) & (
            resid_out[i] > median_out - 10 * std_out))] = 0

    mask_in[i] = in_tmp
    mask_mid[i] = mid_tmp
    mask_out[i] = out_tmp
    # if stacked['info'].data['plateifu'][i] == '8997-12704':
    #    break
stacked['mask_in'].data = np.array(mask_in)
stacked['mask_mid'].data = np.array(mask_mid)
stacked['mask_out'].data = np.array(mask_out)

# mask_fixed_in = fits.ImageHDU(np.array(mask_in), name='mask_in')
# mask_fixed_mid = fits.ImageHDU(np.array(mask_mid), name='mask_mid')
# mask_fixed_out = fits.ImageHDU(np.array(mask_out), name='mask_out')
# stacked.append(mask_fixed_in)
# stacked.append(mask_fixed_mid)
# stacked.append(mask_fixed_out)

stacked.writeto(file_path + 'stacked_sigfix_mask.fits', overwrite=True)
