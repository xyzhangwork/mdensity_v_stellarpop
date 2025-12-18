import numpy as np
from astropy.io import fits
import warnings
from scipy import stats


def ks_err(split_type):
    parameters = fits.open('../data/parameters_reff2_2.fits')
    split = fits.open('../data/sample_split_outliers_total2.fits')

    gals = parameters[1].data['plateifu']

    total_mass = parameters[1].data['mass_sga_rc15']
    sigma_cen = parameters[1].data['sigma_cen']
    sigma_cen_noise = parameters[1].data['sigma_cen_noise']

    ifus0, _, idx0 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type] == 0],
                                    gals,
                                    return_indices=True)
    ifus1, _, idx1 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type] == 1],
                                    gals,
                                    return_indices=True)
    ks = {'mtot': [], 'sig': []}
    for i in range(1000):
        m_tmp = total_mass + np.random.normal(scale=0.1, size=len(total_mass))
        mks_tmp = stats.ks_2samp(m_tmp[idx0], m_tmp[idx1])
        ks['mtot'].append(mks_tmp.pvalue)
        sigma_tmp = np.array([sigma_cen[j] + np.random.normal(scale=sigma_cen_noise[j]) for j in range(len(sigma_cen))])
        sigks_tmp = stats.ks_2samp(sigma_tmp[idx0], sigma_tmp[idx1])
        ks['sig'].append(sigks_tmp.pvalue)
    return ks
