import astropy.io.fits as fits
import numpy as np
import astropy.constants
from smoothing import smoothspec

if __name__ == '__main__':
    file_path = '/home/xiaoya/sample_select/'
    smoothed_spec = fits.open(file_path + 'stacked_sigfix_mask.fits')
    C = astropy.constants.c.to('km/s').value
    stacked = fits.open(file_path + 'stacked_spec.fits')
    flux_in, flux_mid, flux_out = [], [], []
    ivar_in, ivar_mid, ivar_out = [], [], []
    sres_in, sres_mid, sres_out = [], [], []
    std_in, std_mid, std_out = [], [], []
    mask_in, mask_mid, mask_out = [], [], []

    wave = stacked['WAVE'].data
    dlambda = 5000 * 300 / C
    target_sigma = dlambda * C / wave

    f_in, f_mid, f_out = stacked['flux_in'].data, stacked['flux_mid'].data, stacked['flux_out'].data
    i_in, i_mid, i_out = stacked['ivar_in'].data, stacked['ivar_mid'].data, stacked['ivar_out'].data
    s_in, s_mid, s_out = stacked['std_in'].data, stacked['std_mid'].data, stacked['std_out'].data
    r_in, r_mid, r_out = stacked['sres_in'].data, stacked['sres_mid'].data, stacked['sres_out'].data
    m_in, m_mid, m_out = smoothed_spec['mask_in'].data, smoothed_spec['mask_mid'].data, smoothed_spec['mask_out'].data

    ifu_array = []

    for j in range(len(f_in)):
        if np.nansum(f_in[j]) == 0 or np.nansum(f_mid[j]) == 0 or np.nansum(f_out[j]) == 0:
            print('no spec, bitch', stacked['info'].data['plateifu'][j])
            continue
        if smoothed_spec['info'].data['plateifu'].__contains__(stacked['info'].data['plateifu'][j]):
            idx = np.where(smoothed_spec['info'].data['plateifu'] == stacked['info'].data['plateifu'][j])[0][0]
            mtmp_in, mtmp_mid, mtmp_out = m_in[idx], m_mid[idx], m_out[idx]
        else:
            mtmp_in, mtmp_mid, mtmp_out = np.ones_like(f_in[j]), np.ones_like(f_mid[j]), np.ones_like(f_out[j])
        mask_in.append(mtmp_in)
        mask_mid.append(mtmp_mid)
        mask_out.append(mtmp_out)

        # --------------------- inner ------------------------

        spectra = np.copy(f_in[j])
        error = np.copy(1 / np.sqrt(i_in[j]))
        std = np.copy(s_in[j])
        sres = np.copy(r_in[j])
        # spectra[mask_in==0]=np.nan
        # error[mask_in==0] = np.nan
        # std[mask_in==0] = np.nan

        smooth_spectra = smoothspec(wave, spectra, resolution=target_sigma, inres=C / (2.355 * sres), outwave=wave,
                                    fftsmooth=False, smoothtype='vel')
        smooth_error = smoothspec(wave, error, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres),
                                  outwave=wave,
                                  fftsmooth=False)
        smooth_std = smoothspec(wave, std, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres), outwave=wave,
                                fftsmooth=False)

        flux_in.append(np.copy(smooth_spectra))
        ivar_in.append(np.copy(1 / np.square(smooth_error)))
        std_in.append(np.copy(smooth_std))

        # --------------------- middle ------------------------

        spectra = np.copy(f_mid[j])
        error = np.copy(1 / np.sqrt(i_mid[j]))
        std = np.copy(s_mid[j])
        sres = np.copy(r_mid[j])
        # spectra[mask_mid==0]=np.nan
        # error[mask_mid==0] = np.nan
        # std[mask_mid==0] = np.nan

        smooth_spectra = smoothspec(wave, spectra, resolution=target_sigma, inres=C / (2.355 * sres), outwave=wave,
                                    fftsmooth=False)
        smooth_error = smoothspec(wave, error, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres),
                                  outwave=wave,
                                  fftsmooth=False)
        smooth_std = smoothspec(wave, std, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres), outwave=wave,
                                fftsmooth=False)

        flux_mid.append(np.copy(smooth_spectra))
        ivar_mid.append(np.copy(1 / np.square(smooth_error)))
        std_mid.append(np.copy(smooth_std))

        # --------------------- out ------------------------

        spectra = np.copy(f_out[j])
        error = np.copy(1 / np.sqrt(i_out[j]))
        std = np.copy(s_out[j])
        sres = np.copy(r_out[j])
        # spectra[mask_out==0]=np.nan
        # error[mask_out==0] = np.nan
        # std[mask_out==0] = np.nan

        smooth_spectra = smoothspec(wave, spectra, resolution=target_sigma, inres=C / (2.355 * sres), outwave=wave,
                                    fftsmooth=False)
        smooth_error = smoothspec(wave, error, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres),
                                  outwave=wave,
                                  fftsmooth=False)
        smooth_std = smoothspec(wave, std, resolution=target_sigma / np.sqrt(2), inres=C / (2.355 * sres), outwave=wave,
                                fftsmooth=False)

        flux_out.append(np.copy(smooth_spectra))
        ivar_out.append(np.copy(1 / np.square(smooth_error)))
        std_out.append(np.copy(smooth_std))
        ifu_array.append(stacked[1].data[j][0])

    primary_hdu = fits.PrimaryHDU(np.zeros(2))
    wave = fits.ImageHDU(wave, name='wave')
    plateifus = fits.Column(name='plateifu', array=np.array(ifu_array), format='32A')
    cols = fits.ColDefs([plateifus])
    pltifu = fits.BinTableHDU.from_columns(cols, name='info')

    flux_fixed_in = fits.ImageHDU(np.array(flux_in), name='flux_in')
    flux_fixed_mid = fits.ImageHDU(np.array(flux_mid), name='flux_mid')
    flux_fixed_out = fits.ImageHDU(np.array(flux_out), name='flux_out')

    ivar_fixed_in = fits.ImageHDU(np.array(ivar_in), name='ivar_in')
    ivar_fixed_mid = fits.ImageHDU(np.array(ivar_mid), name='ivar_mid')
    ivar_fixed_out = fits.ImageHDU(np.array(ivar_out), name='ivar_out')

    mask_fixed_in = fits.ImageHDU(np.array(mask_in), name='mask_in')
    mask_fixed_mid = fits.ImageHDU(np.array(mask_mid), name='mask_mid')
    mask_fixed_out = fits.ImageHDU(np.array(mask_out), name='mask_out')

    try:
        np.save('std_in', np.array(std_in))
        np.save('std_mid', np.array(std_mid))
        np.save('std_out', np.array(std_out))
        std_fixed_in = fits.ImageHDU(np.array(std_in), name='std_in')
        std_fixed_mid = fits.ImageHDU(np.array(std_mid), name='std_mid')
        std_fixed_out = fits.ImageHDU(np.array(std_out), name='std_out')
        hdul = fits.HDUList([primary_hdu, pltifu, wave,
                             flux_fixed_in, flux_fixed_mid, flux_fixed_out,
                             ivar_fixed_in, ivar_fixed_mid, ivar_fixed_out,
                             std_fixed_in, std_fixed_mid, std_fixed_out,
                             mask_fixed_in, mask_fixed_mid, mask_fixed_out])
    except:
        hdul = fits.HDUList([primary_hdu, pltifu, wave,
                             flux_fixed_in, flux_fixed_mid, flux_fixed_out,
                             ivar_fixed_in, ivar_fixed_mid, ivar_fixed_out,
                             mask_fixed_in, mask_fixed_mid, mask_fixed_out])

    hdul.writeto('stacked_smooth_sigma300_mask.fits', overwrite=True)
