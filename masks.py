from astropy.io import fits
import astropy.cosmology as cosmology
import math
import numpy as np

cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)
mask = fits.open('./data/sample_mask_new.fits')
masking = fits.open('./data/masking.fits')
manga_sga = fits.open('./data/manga_sga_z.fits')

file_path = '/Volumes/ZXYwork/new_stack/'
sga_28 = fits.open(file_path + 'sga_mass_new_28mag.fits')
mask[1].data['cen<psf'] = np.full(len(mask[1].data['cen<psf']), 1)
mask[1].data['mass11.2'] = np.full(len(mask[1].data['mass11.2']), 1)
mask[1].data['outliers_mlcr'] = np.full(len(mask[1].data['outliers_mlcr']), 0)
mask[1].data['negative_mass'] = np.full(len(mask[1].data['negative_mass']), 0)
mask[1].data['reobserved'] = np.full(len(mask[1].data['reobserved']), 0)
mask[1].data['small_pixnum'] = np.full(len(mask[1].data['small_pixnum']), 0)

_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           sga_28[1].data['plateifu'][sga_28[1].data['mass_sga_rc15'] > 11.2],
                           return_indices=True)
mask[1].data['mass11.2'][idx] = 0
_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           sga_28[1].data['plateifu'][sga_28[1].data['Re_arc'] / 2 > 2.5],
                           return_indices=True)
mask[1].data['cen<psf'][idx] = 0

outliers_mlcr = np.array(['10495-6101', '10505-6102', '10842-9102', '11006-6103',
                          '11017-12702', '11745-12701', '11962-12702', '12073-12705',
                          '12087-3702', '12495-12705', '12685-6101', '7978-12701',
                          '7981-12705', '8097-6101', '8131-3701', '8143-6103',
                          '8256-12704', '8274-12704', '8449-6103', '8451-12701',
                          '8466-3703', '8601-3702', '8614-6103', '9033-3701',
                          '9488-6104'])
_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           outliers_mlcr,
                           return_indices=True)
mask[1].data['outliers_mlcr'][idx] = 1

negative_mass = np.array(['11758-12705'])
_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           negative_mass,
                           return_indices=True)
mask[1].data['negative_mass'][idx] = 1

visual = np.array(['8601-3702', '9087-6103'])
_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           visual,
                           return_indices=True)
mask[1].data['outliers_visual'][idx] = 1

reobs = np.array(['12667-1902', '12675-3702', '8274-6103', '8274-6104', '8333-12704',
                  '8451-6102', '8451-6103', '8456-6104', '8651-9102', '9036-1901',
                  '9036-3703', '9884-1902'])
_, idx, _ = np.intersect1d(mask[1].data['plateifu'],
                           reobs,
                           return_indices=True)
mask[1].data['reobserved'][idx] = 1
gals = mask[1].data['plateifu'][(mask[1].data['no_sga_ellp'] == 0) & (mask[1].data['no_reff'] == 0) &
                                (mask[1].data['no_sga_prof'] == 0) & (mask[1].data['no_maps_url'] == 0) &
                                (mask[1].data['no_sga_url'] == 0) & (mask[1].data['ba0.3'] == 0) &
                                (mask[1].data['negative_mass'] == 0) & (mask[1].data['mass11.2'] == 0) & (
                                            mask[1].data['outliers_mlcr'] == 0) &
                                (mask[1].data['outliers_visual'] == 0) &
                                (mask[1].data['cen<psf'] == 0) &
                                #(mask[1].data['small_pixnum'] == 0) &
                                (mask[1].data['morph'] == 0) & (mask[1].data['outliers_psf'] == 0) &
                                (mask[1].data['reobserved'] == 0)]
for gal in gals:
    maps_path = 'maps path'
    idx_ml = np.where(sga_28[1].data['plateifu'] == gal)[0][0]
    idx_sga = np.where(manga_sga[1].data['plateifu'] == gal)[0][0]
    idx_mask = np.where(mask[1].data['plateifu'] == gal)[0][0]

    z_sga = manga_sga[1].data['redshift'][idx_sga]

    hdap = fits.open(maps_path + 'manga-' + gal + '-MAPS-VOR10-MILESHC-MASTARSSP.fits.gz')
    d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value
    reff = sga_28[1].data['Re_kpc'][idx_ml]
    r_center = reff / 2

    ba = sga_28[1].data['ba'][idx_ml]
    ag = sga_28[1].data['pa'][idx_ml]
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
    A = A * (d_a * np.pi / (3600 * 180))

    A[np.isnan(hdap['SPX_SNR'].data)] = np.nan

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

    if masking['info'].data['plateifu'].__contains__(gal):
        mask_snr = mask_snr & (masking[gal].data == 0)
    if np.nansum(((bining == 1) & mask_snr) * 1) < 20 or np.nansum(((bining == 2) & mask_snr) * 1) < 20 or np.nansum(
            ((bining == 3) & mask_snr) * 1) < 20:
        mask[1].data['small_pixnum'][idx_mask] = 1
mask.writeto('sample_mask_new.fits',overwrite=True)