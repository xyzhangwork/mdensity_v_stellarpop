import numpy as np
from astropy.io import fits
from smoothing import smoothspec
import astropy.constants
from scipy import interpolate
import pyphot
from pyphot import unit
import pandas as pd

l = pyphot.LickLibrary()
index_names = np.array(['CN_1', 'CN_2', 'Ca4227', 'Fe4383', 'Ca4455',
                        'Fe4531', 'Fe5015', 'Mg_b', 'Fe5270', 'Fe5335',
                        'Fe5406', 'Fe5709', 'Fe5782', 'MgFe', 'Mgb/Fe'])
ssp_names = []
ssp_path = 'SSP path/IMF type'
afe = ['aFem02', 'aFep00', 'aFep02', 'aFep04', 'aFep06']
met = ['Zm1.79', 'Zm1.49', 'Zm1.26', 'Zm0.96', 'Zm0.66',
       'Zm0.35', 'Zm0.25', 'Zp0.06', 'Zp0.15', 'Zp0.26']
iso = ['_iTp0.00', '_iTp0.00', '_iTp0.00', '_iTp0.40', '_iTp0.40']
age = np.array(['T00.0300', 'T00.0400', 'T00.0500', 'T00.0600', 'T00.0700', 'T00.0800',
                'T00.0900', 'T00.1000', 'T00.1500', 'T00.2000', 'T00.2500', 'T00.3000',
                'T00.3500', 'T00.4000', 'T00.4500', 'T00.5000', 'T00.6000', 'T00.7000',
                'T00.8000', 'T00.9000', 'T01.0000', 'T01.2500', 'T01.5000', 'T01.7500',
                'T02.0000', 'T02.2500', 'T02.5000', 'T02.7500', 'T03.0000', 'T03.2500',
                'T03.5000', 'T03.7500', 'T04.0000', 'T04.5000', 'T05.0000', 'T05.5000',
                'T06.0000', 'T06.5000', 'T07.0000', 'T07.5000', 'T08.0000', 'T08.5000',
                'T09.0000', 'T09.5000', 'T10.0000', 'T10.5000', 'T11.0000', 'T11.5000',
                'T12.0000', 'T12.5000', 'T13.0000', 'T13.5000', 'T14.0000'])
wavelengths = np.linspace(3540.5, 7409.6, num=4300)
wave_mask = wavelengths > 3622
C = astropy.constants.c.to('km/s').value
stacked = fits.open('./data/stacked_spec.fits')
wave = stacked['WAVE'].data
dlambda = 5000 * 300 / C
target_sigma = dlambda * C / wave
func_res = interpolate.interp1d(wave, target_sigma)
target_res = func_res(wavelengths[wave_mask])
fwhm = 2.5
inres = C * fwhm / (2.355 * wavelengths[wave_mask])
# inres=C/(2.355*R) --vel
# R=ckms / fwhm=ckms/(sigma * sigma_to_fwhm)
#
data = {}
for name in index_names:
    data[name] = []
if __name__ == '__main__':
    for i in range(len(afe)):
        for j in range(len(met)):
            for k in range(len(age)):
                ssp_names.append('Mbi3.50' + met[j] + age[k] + iso[i] + '_ACFep00_' + afe[i])
                file_path = ssp_path + afe[i] + '/Mbi3.50' + met[j] + age[k] + iso[i] + '_ACFep00_' + afe[i] + '.fits'
                flux_ssp = fits.getdata(file_path, 0)
                smoothed_ssp = smoothspec(wavelengths[wave_mask], flux_ssp[wave_mask],
                                          resolution=target_res, inres=inres,
                                          outwave=wavelengths[wave_mask],
                                          fftsmooth=False, smoothtype='vel')
                for name in index_names[:-2]:
                    fk = l[name]
                    data[name].append(fk.get(wavelengths[wave_mask] * unit('AA'), smoothed_ssp, axis=1))
                data['MgFe'].append(np.sqrt(data['Mg_b'][-1] * (0.72 * data['Fe5270'][-1] + 0.28 * data['Fe5335'][-1])))
                data['Mgb/Fe'].append(data['Mg_b'][-1] / (0.5 * (data['Fe5270'][-1] + data['Fe5335'][-1])))

    df = pd.DataFrame(data=data, index=ssp_names)
    df.to_csv('./data/IMF type_ssp.csv')


