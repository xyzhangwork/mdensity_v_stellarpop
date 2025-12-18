# coding: utf-8
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse
import sep
from photutils.segmentation import SegmentationImage
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources

gals_masking = np.array(['10517-12705', '10519-12703', '10843-12701', '10843-12705',
                         '11745-1901', '11828-12701', '11947-12702', '11950-12703',
                         '11950-1902', '11950-9102', '12068-6101', '12093-12702',
                         '12495-3704', '12675-12705', '12675-6101', '8087-6102',
                         '8239-12701', '8248-9101', '8440-12702', '8455-12703',
                         '8484-12701', '8488-6102', '8592-12703', '8592-12705', '8617-1901',
                         '8624-12701', '8714-12701', '8717-1901', '8717-6104', '8722-12701',
                         '8725-12704', '8725-6104', '8947-6101', '8948-6104', '8979-12702',
                         '8980-12702', '8989-12704', '8995-12703', '9001-12701',
                         '9043-12701', '9094-12702', '9865-12703'])
plateifus = fits.Column(name='plateifu', array=gals_masking, format='32A')
cols = fits.ColDefs([plateifus])
pltifu = fits.BinTableHDU.from_columns(cols, name='info')
primary_hdu = fits.PrimaryHDU(np.zeros(2))
path = './data/'
masks = [primary_hdu, pltifu]
sga_mass = fits.open('./data/sga_mass_new_28mag2.fits')
for i, cat in enumerate(gals_masking):
    plate = cat[0:cat.find('-')]
    ifu = cat[cat.find('-') + 1:]
    idx_sga = np.where(sga_mass[1].data['plateifu'] == cat)[0][0]

    maps_path = 'maps path'
    logcube_path = 'logcubes path'

    spx = fits.open(maps_path + 'manga-' + str(plate) + '-' + str(ifu) + '-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz')

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    colors = ['', 'red', 'green', 'blue']

    for i in range(1, 4):
        e = Ellipse(xy=(xc, yc),
                    width=re * i,
                    height=re * i * ba,
                    angle=90 + ag)
        e.set_facecolor('none')
        e.set_edgecolor(colors[i])
        ax.add_artist(e)
    m, s = np.mean(data_sub), np.std(data_sub)
    im = ax.imshow(data_sub, interpolation='nearest', cmap='Greys',
                   vmin=m - s, vmax=m + s, origin='lower')

    #'''

    segment_map = detect_sources(data_sub, bkg.globalrms, npixels=10)
    if not segment_map:
        print('no sources found', cat)
    else:
        segm = SegmentationImage(segment_map.data)
        segm_deblend = deblend_sources(data_sub, segm, npixels=10, nlevels=32, contrast=0.001, progress_bar=False)
        mask_tmp = np.copy(segm_deblend.data)
        mask_tmp[mask_tmp == mask_tmp[int(yc), int(xc)]] = 0
        #masks.append(fits.ImageHDU(mask_tmp, name=cat))
        ax.imshow(segm_deblend.data, origin='lower', alpha=0.4)
    # plot an ellipse for each object

    # ax.imshow(bining, origin='lower', alpha=0.4)

    # fig.show()
    # fig.tight_layout()
    fig.savefig(path + cat + '_missing.png')

hdul = fits.HDUList(masks)
hdul.writeto('masking.fits', overwrite=True)
