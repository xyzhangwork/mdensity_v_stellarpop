from astropy.io import fits
import numpy as np
from astropy.table import Table

if __name__ == '__main__':
    parameters = fits.open('parameters_spx2_max_reff2.fits')
    morph_mask = (parameters[1].data['gzoo'][:, 0] == 1) | (parameters[1].data['ttype'] > 0)

    sga_mass = fits.open('sga_mass_new.fits')
    lumes = ['lum10kpc', 'lumout(max-20kpc)']
    masses = ['mass10kpc', 'massout(max-20kpc)']
    sample_bin = [[1, 35], [65, 99]]
    catogory = {'lum': np.full(len(parameters[1].data), np.nan),
                'mass': np.full(len(parameters[1].data), np.nan),
                'sigma': np.full(len(parameters[1].data), np.nan),
                'mtot': np.full(len(parameters[1].data), np.nan),
                'sigma_mtot': np.full(len(parameters[1].data), np.nan)}
    ifus = parameters[1].data['plateifu'][~morph_mask]
    _, idp, _ = np.intersect1d(parameters[1].data['plateifu'],
                               ifus,
                               return_indices=True)
    _, idm, _ = np.intersect1d(sga_mass[1].data['plateifu'],
                               ifus,
                               return_indices=True)

    dx = parameters[1].data[idp][lumes[0]]
    dy = parameters[1].data[idp][lumes[1]]
    mask_nan = (~np.isnan(dx)) & (~np.isnan(dy))
    z = np.polyfit(dx[mask_nan], dy[mask_nan], 1)
    p = np.poly1d(z)
    var = np.zeros_like(dx)
    var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]
    var[~mask_nan] = np.nan
    catogory_tmp = np.full_like(catogory['lum'][idp], np.nan)
    for i in range(2):
        mask_tmp = (var <= np.percentile(var[mask_nan], sample_bin[i][1])) & \
                   (var >= np.percentile(var[mask_nan], sample_bin[i][0]))
        catogory_tmp[mask_tmp] = i  # + type_i * 2
        print(np.percentile(var[mask_nan], sample_bin[i][1]), np.percentile(var[mask_nan], sample_bin[i][0]))
    catogory['lum'][idp] = catogory_tmp.copy()

    dx = parameters[1].data[idp][masses[0]]
    dy = parameters[1].data[idp][masses[1]]
    mask_nan = (~np.isnan(dx)) & (~np.isnan(dy)) & (~np.isinf(dx)) & (~np.isinf(dy))
    z = np.polyfit(dx[mask_nan], dy[mask_nan], 1)
    p = np.poly1d(z)
    var = np.zeros_like(dx)
    var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]
    var[~mask_nan] = np.nan
    catogory_tmp = np.full_like(catogory['mass'][idp], np.nan)
    for i in range(2):
        mask_tmp = (var <= np.percentile(var[mask_nan], sample_bin[i][1])) & \
                   (var >= np.percentile(var[mask_nan], sample_bin[i][0]))
        catogory_tmp[mask_tmp] = i
        print(np.percentile(var[mask_nan], sample_bin[i][1]), np.percentile(var[mask_nan], sample_bin[i][0]))
    catogory['mass'][idp] = catogory_tmp.copy()

    # ------------------------------ sigma splitting ------------------------------------

    dy = np.log10(parameters[1].data[idp]['sigma_cen'])
    dx = parameters[1].data[idp][masses[0]]
    idx = np.arange(len(dx))
    noise = parameters[1].data[idp]['sigma_cen_noise']
    mask_nan = (~np.isnan(dx)) & (~np.isnan(dy)) & (~np.isinf(dx)) & (~np.isinf(dy)) & (~np.isinf(noise)) & (
        ~np.isnan(noise))  # &(dy>np.nanpercentile(dy,5))
    idx = idx[mask_nan]

    pivot = np.nanmean(dy)
    var = np.full_like(dx, np.nan)

    p0 = np.polyfit(dx[mask_nan], dy[mask_nan], deg=1)
    p = np.poly1d(p0)
    var[idx] = p(dx[idx]) - dy[idx]

    catogory_tmp = np.full_like(catogory['sigma'][idp], np.nan)
    # i=0: higher; i=1: lower
    for i in range(2):
        mask_nan = ~np.isnan(var)
        mask_tmp = (var <= np.percentile(var[mask_nan], sample_bin[i][1])) & \
                   (var >= np.percentile(var[mask_nan], sample_bin[i][0]))
        catogory_tmp[mask_tmp] = i
        print(np.percentile(var[mask_nan], sample_bin[i][1]), np.percentile(var[mask_nan], sample_bin[i][0]))
    catogory['sigma'][idp] = catogory_tmp.copy()

    # ------------------------------ total mass -------------------------------------
    dx = sga_mass[1].data['mass_sga_rc15'][idm]
    dy = parameters[1].data[idp][masses[0]]
    mask_nan = (~np.isnan(dx)) & (~np.isnan(dy)) & (~np.isinf(dx)) & (~np.isinf(dy))
    z = np.polyfit(dx[mask_nan], dy[mask_nan], 1)
    p = np.poly1d(z)
    var = np.zeros_like(dx)
    var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]
    var[~mask_nan] = np.nan
    catogory_tmp = np.full_like(catogory['mtot'][idp], np.nan)
    for i in range(2):
        mask_tmp = (var <= np.percentile(var[mask_nan], sample_bin[i][1])) & \
                   (var >= np.percentile(var[mask_nan], sample_bin[i][0]))
        catogory_tmp[mask_tmp] = i
        print(np.percentile(var[mask_nan], sample_bin[i][1]), np.percentile(var[mask_nan], sample_bin[i][0]))
    catogory['mtot'][idp] = catogory_tmp.copy()
    # ------------------------------ sigma_cen_v_total mass -------------------------------------
    dx = np.log10(parameters[1].data[idp]['sigma_cen'])
    dy = sga_mass[1].data['mass_sga_rc15'][idm]
    mask_nan = (~np.isnan(dx)) & (~np.isnan(dy)) & (~np.isinf(dx)) & (~np.isinf(dy))
    z = np.polyfit(dx[mask_nan], dy[mask_nan], 1)
    p = np.poly1d(z)
    var = np.zeros_like(dx)
    var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]
    var[~mask_nan] = np.nan
    catogory_tmp = np.full_like(catogory['sigma_mtot'][idp], np.nan)
    for i in range(2):
        mask_tmp = (var <= np.percentile(var[mask_nan], sample_bin[i][1])) & \
                   (var >= np.percentile(var[mask_nan], sample_bin[i][0]))
        catogory_tmp[mask_tmp] = i
        print(np.percentile(var[mask_nan], sample_bin[i][1]), np.percentile(var[mask_nan], sample_bin[i][0]))
    catogory['sigma_mtot'][idp] = catogory_tmp.copy()

    t = Table([parameters[1].data['plateifu'],
               catogory['lum'],
               catogory['mass'],
               catogory['sigma'],
               catogory['mtot'], catogory['sigma_mtot']],
              names=['plateifu',
                     'l10kpc_v_l20kpc',
                     'm10kpc_v_m20kpc',
                     'm10kpc_v_sigmacen',
                     'mtot_v_m10kpc', 'sigma_v_mtot'])
    t.write('sample_split_outliers_total.fits', overwrite=True)
