import numpy as np
import requests
import astropy.io.fits as fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy


def filter_healpix_mask(mask_hp, cat, ra='ra', dec='dec', verbose=True):
    """Filter a catalog through a Healpix mask.
    Parameters
    ----------
    mask_hp : healpy mask file
        healpy mask file
    cat : numpy array or astropy.table
        Catalog that includes the coordinate information
    ra : string
        Name of the column for R.A.
    dec : string
        Name of the column for Dec.
    verbose : boolen, optional
        Default: True
    Return
    ------
        Selected objects that are covered by the mask.
    """
    # Read the healpix mask
    mask = healpy.read_map(mask_hp, nest=True, dtype=np.bool)

    nside, hp_indices = healpy.get_nside(mask), np.where(mask)[0]
    phi, theta = np.radians(cat[ra]), np.radians(90. - cat[dec])
    hp_masked = healpy.ang2pix(nside, theta, phi, nest=True)
    select = np.in1d(hp_masked, hp_indices)

    if verbose:
        print("# %d/%d objects are selected by the mask" % (select.sum(), len(cat)))

    return cat[select]


if __name__ == "__main__":
    # test = 'https://www.legacysurvey.org/viewer/fits-cutout?ra=185.8210&dec=13.0487&pixscale=0.5&layer=ls-dr9&size=70'
    url0 = "https://www.legacysurvey.org/viewer/fits-cutout?ra="
    url1 = "&dec="
    url2 = "&pixscale=0.5&layer=decals-dr7&size=100"

    hsc_mask = fits.open('s20a_fdfc_hp_contarea_izy-gt-5.fits')

    file_drpall = '/home/xiaoya/sample_select/drpall-v3_1_1.fits'
    drpall = fits.getdata('/Users/evrinezhang/Desktop/drpall-v3_1_1.fits', 1)
    """

    '''
    MaNGA v. DECaLS
    '''
    plateifu, objra, objdec = [], [], []
    n=0
    mask=(drpall['objdec']>30)|(drpall['objdec']<-30)
    for manga_info in drpall[mask]:
        if n>10:
            break
        ra = manga_info['objra']
        dec = manga_info['objdec']
        url = url0 + str(round(ra, 4)) + url1 + str(round(dec, 4)) + url2
        response = requests.head(url)
        print(manga_info['plateifu'])
        if response.status_code == 200:
            plateifu.append(manga_info['plateifu'])
            objra.append(manga_info['objra'])
            objdec.append(manga_info['objdec'])
            print(manga_info['objdec'], manga_info['objra'])
            n += 1
    #t = Table([plateifu, objra, objdec], names=['plateifu', 'objra', 'objdec'])
    #t.write('manga_decals_ds9.fits', overwrite=True)

    '''
    MaNGA v. HSC
    '''
    cat_manga_hsc = filter_healpix_mask(hsc_mask, drpall, ra='objra', dec='objdec')
    # manga_decals_ds9.append([manga_info['plateifu'], manga_info['objra'], manga_info['objdec']])
    t = Table([cat_manga_hsc['plateifu'], cat_manga_hsc['objra'], cat_manga_hsc['objdec']],
              names=['plateifu', 'objra', 'objdec'])
    t.write('manga_hsc.fits', overwrite=True)
    """

    sga = fits.getdata('/Users/evrinezhang/Desktop/SGA-2020.fits', 1)
    cat_ra = sga['ra']
    cat_dec = sga['dec']
    cat_z = sga['Z_LEDA']

    drp_ra = drpall['objra']
    drp_dec = drpall['objdec']
    drp_z = drpall['z']

    cat_coo = SkyCoord(ra=cat_ra * u.degree, dec=cat_dec * u.degree)
    drp_coo = SkyCoord(ra=drp_ra * u.degree, dec=drp_dec * u.degree)
    idxdrp, idxcat, d2d, d3d = cat_coo.search_around_sky(drp_coo, 3 * u.arcsec)

    cat_z = cat_z[idxcat]
    drp_z = drp_z[idxdrp]
    valid = abs(cat_z - drp_z) <= 0.0005

    plateifu, objra, objdec, m, redshift, sgaid = drpall['plateifu'][idxdrp], \
                                                  drpall['objra'][idxdrp], \
                                                  drpall['objdec'][idxdrp], \
                                                  np.log10(drpall['nsa_elpetro_mass'][idxdrp] / 0.7 ** 2), \
                                                  drpall['z'][idxdrp], \
                                                  sga['sga_id'][idxcat]

    t = Table([plateifu[valid],sgaid[valid], objra[valid], objdec[valid], m[valid], redshift[valid]],
              names=['plateifu', 'sgaid', 'objra', 'objdec', 'm', 'redshift'])
    t.write('manga_sga_z.fits', overwrite=True)
