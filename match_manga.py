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
    file_drpall = './data/drpall-v3_1_1.fits'
    drpall = fits.getdata('./data/drpall-v3_1_1.fits', 1)

    sga = fits.getdata('./data/SGA-2020.fits', 1)
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
