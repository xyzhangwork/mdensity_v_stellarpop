# coding: utf-8
from astropy.io import fits
import numpy as np
import math
import sep
from photutils.segmentation import SegmentationImage
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
import os

parameters = fits.open('../data/parameters_spx2_max_reff2.fits')
sga_mass = fits.open('../data/sga_mass_new.fits')
mask_hdu = fits.open('../data/sample_mask.fits')
mask_names = ['no_maps',
              'zero_maps_size',
              'no_maps_url',
              'no_sga_url',
              'no_sga_ellp',
              'no_reff',
              'no_sga_prof',
              'cen<psf',
              'mass11.2',
              'outliers_mlcr',
              'ba0.3',
              'negative_mass',
              'outliers_psf']
mask = mask_hdu[1].data[mask_names[0]] != 2
for i in range(len(mask_names)):
    mask = mask & (mask_hdu[1].data[mask_names[i]] == 0)
gals = mask_hdu[1].data['plateifu'][mask]

file = open('gal_mask.txt')
lines = file.readlines()
gals_masking = np.array([lines[i][:lines[i].find('_')] for i in range(len(lines))])

plateifus = fits.Column(name='plateifu', array=gals_masking, format='32A')
cols = fits.ColDefs([plateifus])
pltifu = fits.BinTableHDU.from_columns(cols, name='info')
primary_hdu = fits.PrimaryHDU(np.zeros(2))
masks = [primary_hdu]
for i, cat in enumerate(gals_masking):
    plate = cat[0:cat.find('-')]
    ifu = cat[cat.find('-') + 1:]
    idx_sga = np.where(sga_mass[1].data['plateifu'] == cat)[0][0]

    maps_path = 'manga maps path'
    spx = fits.open(maps_path + 'manga-' + cat + '-MAPS-SPX-MILESHC-MASTARSSP.fits.gz')

    ba = sga_mass[1].data['ba'][idx_sga]
    ag = sga_mass[1].data['pa'][idx_sga]
    s = math.sin(ag * math.pi / 180)
    c = math.cos(ag * math.pi / 180)
    Xc, Yc = spx[0].header['objra'], spx[0].header['objdec']
    xr = spx[1].header['crpix1']
    yr = spx[1].header['crpix2']
    xu = spx[1].header['PC1_1']
    yu = spx[1].header['PC2_2']
    Yr = spx[1].header['crval2']
    Xr = spx[1].header['crval1']
    xc = xr + (Xc - Xr) / xu
    yc = yr + (Yc - Yr) / yu
    D = spx[1].header['NAXIS1']
    pos0 = np.full((D, D), np.arange(0, D))
    pos = np.zeros((D, D, 2))
    pos[:, :, 1] = pos0
    pos[:, :, 0] = pos0.T
    A = np.sqrt((((pos[:, :, 1] - xc) * c + (pos[:, :, 0] - yc) * s) / ba) ** 2 + (
            (pos[:, :, 0] - yc) * c - (pos[:, :, 1] - xc) * s) ** 2) * 0.5

    reff = sga_mass[1].data['Re_arc'][idx_sga]
    bins_in = A <= reff * 0.5
    bins_out = A >= 1 * reff
    bins_mid = (A >= reff * 0.5) & (A <= reff)
    bining = np.full(np.shape(bins_in), -1)
    bining[bins_in] = 1
    bining[bins_mid] = 2
    data0 = spx['SPX_MFLUX'].data * A * A
    data0[bining != 2] = np.nan
    mask = data0 > np.percentile(data0[~np.isnan(data0)], 93)
    bining[mask] = -1
    bining[bins_out] = 3
    bining[~(spx['SPX_SNR'].data >= 3)] = -1

    data = np.copy(spx['SPX_MFLUX'].data)
    data = data.byteswap().newbyteorder()
    bkg = sep.Background(data, bw=int(D / 2), bh=int(D / 2))
    bkg_image = bkg.back()
    bkg_rms = bkg.rms()
    data_sub = data - bkg
    objects = sep.extract(data_sub, 1, err=bkg.globalrms)
    re = reff * 2

    segment_map = detect_sources(data_sub, bkg.globalrms, npixels=10)
    if segment_map:
        segm = SegmentationImage(segment_map.data)
        segm_deblend = deblend_sources(data_sub, segm, npixels=10, nlevels=32, contrast=0.001, progress_bar=False)
        mask_tmp = np.copy(segm_deblend.data)
        mask_tmp[mask_tmp == mask_tmp[int(yc), int(xc)]] = 0
        masks.append(fits.ImageHDU(mask_tmp, name=cat))


hdul = fits.HDUList(masks)
hdul.writeto('masking.fits', overwrite=True)
