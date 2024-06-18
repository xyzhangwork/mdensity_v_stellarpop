import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from os import path
import astropy.constants
from mangadap.util.sampling import Resample, spectral_coordinate_step
import ppxf as ppxf_package
from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.table import Table
import sys
from mangadap.proc.spectralstack import SpectralStack


def clip_outliers(galaxy, bestfit, goodpixels):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma
    in relative error until the bad bins don't change any more.
    """
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels] / np.sum(bestfit[goodpixels] ** 2)
        resid = scale * bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < 3 * err)
        if np.array_equal(goodpixels, ok_old):
            break

    return goodpixels


def fit_and_clean(templates, galaxy, velscale, start, goodpixels0, lam, lam_temp):
    print('##############################################################')
    goodpixels = goodpixels0.copy()
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=4, degree=20, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels, plot=False, trig=True)

    # plt.figure(figsize=(20, 3))
    # plt.subplot(121)
    # pp.plot()

    goodpixels = clip_outliers(galaxy, pp.bestfit, goodpixels)

    # Add clipped pixels to the original masked emission lines regions and repeat the fit
    goodpixels = np.intersect1d(goodpixels, goodpixels0)
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=4, degree=20, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels, plot=False, trig=True)

    # plt.subplot(122)
    # pp.plot()

    optimal_template = templates @ pp.weights

    return pp, optimal_template


if __name__ == '__main__':
    # file_path = '/Users/evrinezhang/Desktop/'
    file_path = '/home/xiaoya/sample_select/'
    C = astropy.constants.c.to('km/s').value
    specstack = SpectralStack()
    # i_num=int(sys.argv[1])
    # i_num = 0#int(sys.argv[1])
    # for i in range(int(i_num), int(i_num + 1)):
    if True:
        # stacked = fits.open(file_path + 'stacked_mass_kpc_new_' + str(i) + '.fits')
        # stacked = fits.open(file_path + 'stacked_' + str(i) + '.fits')
        stacked = fits.open(file_path + 'stacked_spec.fits')
        # stacked = fits.open(file_path + 'stacked_spec_basslow.fits')

        sigmas = {'sigma_in': [[], []], 'sigma_mid': [[], []], 'sigma_out': [[], []]}
        newwave_in, newwave_mid, newwave_out = [], [], []
        flux_in, flux_mid, flux_out = [], [], []
        ivar_in, ivar_mid, ivar_out = [], [], []
        std_in, std_mid, std_out = [], [], []
        sres_in, sres_mid, sres_out = [], [], []
        mask_in, mask_mid, mask_out = [], [], []
        resid_in, resid_mid, resid_out = [], [], []

        wave = stacked['WAVE'].data
        wave_range = [4000, 6000]
        w = (wave > wave_range[0]) & (wave < wave_range[1])
        lam = wave[w]

        velscale = C * np.diff(np.log(lam[-2:]))

        ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
        pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
        FWHM_gal = None  # set this to None to skip convolutiona
        miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])
        stars_templates, ln_lam_temp = miles.templates, miles.ln_lam_temp

        reg_dim = stars_templates.shape[1:]
        stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)

        stars_templates /= np.median(stars_templates)  # Normalizes stellar templates by a scalar
        regul_err = 0.01  # Desired regularization error

        lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
        z = 0
        start = [0, 200.]
        goodpixels = util.determine_goodpixels(np.log(lam), lam_range_temp, z, width=1000)
        ifu_array = []
        errors = {'error_in': [[], []], 'error_mid': [[], []], 'error_out': [[], []]}
        velocities = {'vel_in': [[], []], 'vel_mid': [[], []], 'vel_out': [[], []]}
        # if i_num==0:
        f_in, f_mid, f_out = stacked['flux_in'].data, stacked['flux_mid'].data, stacked['flux_out'].data
        i_in, i_mid, i_out = stacked['ivar_in'].data, stacked['ivar_mid'].data, stacked['ivar_out'].data
        s_in, s_mid, s_out = stacked['std_in'].data, stacked['std_mid'].data, stacked['std_out'].data
        r_in, r_mid, r_out = stacked['sres_in'].data, stacked['sres_mid'].data, stacked['sres_out'].data

        # else:
        #    f_in, f_mid, f_out = stacked['flux_wei_in'].data, stacked['flux_wei_mid'].data, stacked['flux_wei_out'].data
        #    i_in, i_mid, i_out = stacked['ivar_in'].data, stacked['ivar_mid'].data, stacked['ivar_out'].data

        # idx_target = np.where(stacked['info'].data['plateifu'] == '8725-12704')[0][0]
        for j in range(len(f_in)):
            if np.nansum(f_in[j][w]) == 0 or np.nansum(f_mid[j][w]) == 0 or np.nansum(f_out[j][w]) == 0:
                print('no spec, bitch', stacked['info'].data['plateifu'][j])
                continue
            # --------------------- inner ------------------------
            spectra = np.copy(f_in[j])
            error = np.copy(1 / np.sqrt(i_in[j]))
            std = np.copy(s_in[j])
            sres = np.copy(r_in[j])

            galaxy = np.copy(f_in[j][w])
            if np.nansum(np.isnan(galaxy)) >= 1:
                print(j, stacked['plateifu'].data[j])
                continue
            pp, bestfit_template = fit_and_clean(stars_templates, galaxy, velscale, start, goodpixels, lam,
                                                 miles.lam_temp)
            vel, sigma, _, _ = pp.sol
            velocities['vel_in'][0].append(vel)
            wave, spectra, _, _ = specstack.register(wave, np.array([vel]), np.array([spectra]), keep_range=True,
                                                     log=True)
            spectra = spectra[0]

            # alpha=vel+1
            sigmas['sigma_in'][0].append(sigma)
            error_tmp = pp.error * np.sqrt(pp.chi2)
            errors['error_in'][0].append(error_tmp[0])
            errors['error_in'][1].append(error_tmp[1])
            if sigma < 400:
                # '''
                sigma0 = np.sqrt(400 ** 2 - sigma ** 2)
                gauss_std = 2 * np.nanmedian(lam) * sigma0 / C
                g = Gaussian1DKernel(stddev=gauss_std, x_size=41)
                g_err = Gaussian1DKernel(stddev=gauss_std / np.sqrt(2), x_size=41)
                sres_tmp=wave/np.sqrt((wave/sres)**2+gauss_std**2)

                # g = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma0 / C, x_size=41)

                resamp = Resample(spectra, x=wave, e=error, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5, base=10,
                                  newLog=False)
                resamp_std = Resample(spectra, x=wave, e=std, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5,
                                      base=10,
                                      newLog=False)

                # wave=resamp.outx
                spectra = resamp.outy
                error = resamp.oute
                std = resamp_std.oute
                spec_conv = convolve(spectra, g)
                error_conv = convolve(error, g_err)
                std_conv = convolve(std, g_err)

                dw = spectral_coordinate_step(wave, log=True, base=10)
                resamp1 = Resample(spec_conv, x=resamp.outx, e=error_conv, inLog=False,
                                   newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)
                resamp1_std = Resample(spec_conv, x=resamp.outx, e=std_conv, inLog=False,
                                       newRange=[resamp.outx[0], resamp.outx[-1]],
                                       newdx=dw,
                                       base=10, newLog=True)

                newwave_in.append(np.copy(resamp1.outx))
                flux_in.append(np.copy(resamp1.outy))
                ivar_in.append(np.copy(1 / np.square(resamp1.oute)))
                std_in.append(np.copy(resamp1_std.oute))
                sres_in.append(np.copy(sres_tmp[:-1]))

                wave2 = np.copy(resamp1.outx)
                w2 = (wave2 > wave_range[0]) & (wave2 < wave_range[1])
                lam2 = wave2[w2]

                # spec_conv = resamp1.outy
                sigma1 = np.sqrt(500 ** 2 - sigma ** 2)
                g2 = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma1 / C, x_size=41)

                # wave=resamp.outx
                spec_conv1 = convolve(spectra, g2)
                resamp0 = Resample(spectra, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                resamp2 = Resample(spec_conv1, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                velscale = C * np.diff(np.log(lam2[-2:]))
                # goodpixels0 = util.determine_goodpixels(resamp1.outx, lam_range_temp, z, width=1000)
                goodpixels0 = np.arange(len(resamp1.outx[w2]))
                print('new sigma')

                pp0, _ = fit_and_clean(stars_templates, resamp1.outy[w2], velscale, start, goodpixels0, lam2,
                                       miles.lam_temp)
                '''
                newwave_in.append(np.copy(pp0.lam))
                flux_in.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                mask_in.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                '''
                resid0 = resamp0.outy - resamp2.outy
                resid1 = pp0.galaxy - pp0.bestfit
                mask_tmp = (abs(resid0) < np.median(resid0) + 1 * np.std(resid0)) * 1
                mask_in.append(mask_tmp)
                resid_in.append(resid0)
                sigmas['sigma_in'][1].append(pp0.sol[1])
                velocities['vel_in'][1].append(pp0.sol[0])
            else:
                '''
                w2 = (wave > wave_range[0]) & (wave < wave_range[1])

                pp0, _ = fit_and_clean(stars_templates, spectra[w2], velscale, start, goodpixels, lam,
                                       miles.lam_temp)
                newwave_in.append(np.copy(pp0.lam))
                flux_in.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                mask_in.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                '''
                mask_in.append(np.zeros_like(wave[:-1]))
                resid_in.append(np.full_like(wave[:-1], np.nan))

                newwave_in.append(np.copy(wave[:-1]))
                flux_in.append(np.copy(spectra[:-1]))
                ivar_in.append(np.copy(1 / np.square(error[:-1])))
                std_in.append(std)
                sres_in.append(np.copy(sres[:-1]))

                velocities['vel_in'][1].append(np.nan)
                sigmas['sigma_in'][1].append(sigma)

            # --------------------- middle ------------------------
            spectra = np.copy(f_mid[j])
            error = np.copy(1 / np.sqrt(i_mid[j]))
            std = np.copy(s_mid[j])
            sres = np.copy(r_mid[j])
            galaxy = np.copy(f_mid[j][w])
            if np.nansum(np.isnan(galaxy)) >= 1:
                print(j, stacked['plateifu'].data[j])
                continue
            pp, bestfit_template = fit_and_clean(stars_templates, galaxy, velscale, start, goodpixels, lam,
                                                 miles.lam_temp)
            vel, sigma, _, _ = pp.sol
            velocities['vel_mid'][0].append(vel)
            wave, spectra, _, _ = specstack.register(wave, np.array([vel]), np.array([spectra]), keep_range=True,
                                                     log=True)
            spectra = spectra[0]

            # alpha=vel+1
            sigmas['sigma_mid'][0].append(sigma)
            error_tmp = pp.error * np.sqrt(pp.chi2)
            errors['error_mid'][0].append(error_tmp[0])
            errors['error_mid'][1].append(error_tmp[1])
            if sigma < 400:
                # '''
                sigma0 = np.sqrt(400 ** 2 - sigma ** 2)
                gauss_std = 2 * np.nanmedian(lam) * sigma0 / C
                g = Gaussian1DKernel(stddev=gauss_std, x_size=41)
                g_err = Gaussian1DKernel(stddev=gauss_std / np.sqrt(2), x_size=41)
                #new_res = C/wave * np.sqrt((wave/r_mid)**2+gauss_std**2)
                sres_tmp=wave/np.sqrt((wave/sres)**2+gauss_std**2)
                # g = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma0 / C, x_size=41)

                resamp = Resample(spectra, x=wave, e=error, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5, base=10,
                                  newLog=False)
                resamp_std = Resample(spectra, x=wave, e=std, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5,
                                      base=10,
                                      newLog=False)

                # wave=resamp.outx
                spectra = resamp.outy
                std = resamp_std.oute
                error = resamp.oute
                error_conv = convolve(error, g_err)
                std_conv = convolve(std, g_err)
                spec_conv = convolve(spectra, g)
                sres_conv = wave/convolve(wave/sres, g_err)
                dw = spectral_coordinate_step(wave, log=True, base=10)
                resamp1 = Resample(spec_conv, x=resamp.outx, e=error_conv, inLog=False,
                                   newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)
                resamp1_std = Resample(spec_conv, x=resamp.outx, e=std_conv, inLog=False,
                                       newRange=[resamp.outx[0], resamp.outx[-1]],
                                       newdx=dw,
                                       base=10, newLog=True)

                newwave_mid.append(np.copy(resamp1.outx))
                flux_mid.append(np.copy(resamp1.outy))
                ivar_mid.append(np.copy(1 / np.square(resamp1.oute)))
                std_mid.append(np.copy(resamp1_std.oute))
                sres_mid.append(np.copy(sres_tmp[:-1]))
                wave2 = np.copy(resamp1.outx)
                w2 = (wave2 > wave_range[0]) & (wave2 < wave_range[1])
                lam2 = wave2[w2]

                # spec_conv = resamp1.outy
                sigma1 = np.sqrt(500 ** 2 - sigma ** 2)
                g2 = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma1 / C, x_size=41)

                # wave=resamp.outx
                spec_conv1 = convolve(spectra, g2)
                resamp0 = Resample(spectra, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                resamp2 = Resample(spec_conv1, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                velscale = C * np.diff(np.log(lam2[-2:]))
                # goodpixels0 = util.determine_goodpixels(resamp1.outx, lam_range_temp, z, width=1000)
                goodpixels0 = np.arange(len(resamp1.outx[w2]))

                pp0, _ = fit_and_clean(stars_templates, resamp1.outy[w2], velscale, start, goodpixels0, lam2,
                                       miles.lam_temp)
                resid0 = resamp0.outy - resamp2.outy
                resid1 = pp0.galaxy - pp0.bestfit
                mask_tmp = (abs(resid0) < np.median(resid0) + 1 * np.std(resid0)) * 1
                resid_mid.append(resid0)
                mask_mid.append(mask_tmp)
                '''

                newwave_mid.append(np.copy(pp0.lam))
                flux_mid.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                '''

                sigmas['sigma_mid'][1].append(pp0.sol[1])
                velocities['vel_mid'][1].append(pp0.sol[0])
            else:
                '''
                w2 = (wave > wave_range[0]) & (wave < wave_range[1])

                pp0, _ = fit_and_clean(stars_templates, spectra[w2], velscale, start, goodpixels, lam,
                                       miles.lam_temp)
                newwave_mid.append(np.copy(pp0.lam))
                flux_mid.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                mask_mid.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                '''
                mask_mid.append(np.zeros_like(wave[:-1]))
                resid_mid.append(np.full_like(wave[:-1], np.nan))
                newwave_mid.append(np.copy(wave[:-1]))
                flux_mid.append(np.copy(spectra[:-1]))
                ivar_mid.append(np.copy(1 / np.square(error[:-1])))
                sres_mid.append(np.copy(sres[:-1]))
                std_mid.append(std)
                sigmas['sigma_mid'][1].append(sigma)
                velocities['vel_mid'][1].append(np.nan)

            # --------------------- outer ------------------------
            spectra = np.copy(f_out[j])
            error = np.copy(1 / np.sqrt(i_out[j]))
            std = np.copy(s_out[j])
            sres = np.copy(r_out[j])
            galaxy = np.copy(f_out[j][w])
            if np.nansum(np.isnan(galaxy)) >= 1:
                print(j, stacked['plateifu'].data[j])
                continue
            pp, bestfit_template = fit_and_clean(stars_templates, galaxy, velscale, start, goodpixels, lam,
                                                 miles.lam_temp)
            vel, sigma, _, _ = pp.sol
            velocities['vel_out'][0].append(vel)
            wave, spectra, _, _ = specstack.register(wave, np.array([vel]), np.array([spectra]), keep_range=True,
                                                     log=True)
            spectra = spectra[0]

            # alpha=vel+1
            sigmas['sigma_out'][0].append(sigma)
            error_tmp = pp.error * np.sqrt(pp.chi2)
            errors['error_out'][0].append(error_tmp[0])
            errors['error_out'][1].append(error_tmp[1])
            if sigma < 400:
                # '''
                sigma0 = np.sqrt(400 ** 2 - sigma ** 2)
                gauss_std = 2 * np.nanmedian(lam) * sigma0 / C
                g = Gaussian1DKernel(stddev=gauss_std, x_size=41)
                g_err = Gaussian1DKernel(stddev=gauss_std / np.sqrt(2), x_size=41)
                # g = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma0 / C, x_size=41)
                sres_tmp=wave/np.sqrt((wave/sres)**2+gauss_std**2)


                resamp = Resample(spectra, x=wave, e=error, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5, base=10,
                                  newLog=False)
                resamp_std = Resample(spectra, x=wave, e=std, inLog=True, newRange=[wave[0], wave[-1]], newdx=.5,
                                      base=10,
                                      newLog=False)

                # wave=resamp.outx
                spectra = resamp.outy
                error = resamp.oute
                std = resamp_std.oute
                spec_conv = convolve(spectra, g)
                error_conv = convolve(error, g_err)
                std_conv = convolve(std, g_err)

                dw = spectral_coordinate_step(wave, log=True, base=10)
                resamp1 = Resample(spec_conv, x=resamp.outx, e=error_conv, inLog=False,
                                   newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)
                resamp1_std = Resample(spec_conv, x=resamp.outx, e=std_conv, inLog=False,
                                       newRange=[resamp.outx[0], resamp.outx[-1]],
                                       newdx=dw,
                                       base=10, newLog=True)

                newwave_out.append(np.copy(resamp1.outx))
                flux_out.append(np.copy(resamp1.outy))
                ivar_out.append(np.copy(1 / np.square(resamp1.oute)))
                sres_out.append(np.copy(sres_tmp[:-1]))
                wave2 = np.copy(resamp1.outx)
                w2 = (wave2 > wave_range[0]) & (wave2 < wave_range[1])
                lam2 = wave2[w2]
                sigma1 = np.sqrt(500 ** 2 - sigma ** 2)
                g2 = Gaussian1DKernel(stddev=2 * np.nanmedian(lam) * sigma1 / C, x_size=41)

                # wave=resamp.outx
                spec_conv1 = convolve(spectra, g2)
                resamp0 = Resample(spectra, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                resamp2 = Resample(spec_conv1, x=resamp.outx, inLog=False, newRange=[resamp.outx[0], resamp.outx[-1]],
                                   newdx=dw,
                                   base=10, newLog=True)

                # spec_conv = resamp1.outy

                velscale = C * np.diff(np.log(lam2[-2:]))
                #goodpixels0 = util.determine_goodpixels(resamp1.outx, lam_range_temp, z, width=1000)
                goodpixels0 = np.arange(len(resamp1.outx[w2]))

                pp0, _ = fit_and_clean(stars_templates, resamp1.outy[w2], velscale, start, goodpixels0, lam2,
                                       miles.lam_temp)
                resid0 = resamp0.outy - resamp2.outy
                resid1 = pp0.galaxy - pp0.bestfit
                mask_tmp = (abs(resid0) < np.median(resid0) + 1 * np.std(resid0)) * 1
                resid_out.append(resid0)
                std_out.append(np.copy(resamp1_std.oute))
                mask_out.append(mask_tmp)
                '''
                newwave_out.append(np.copy(pp0.lam))
                flux_out.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                mask_out.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                '''

                sigmas['sigma_out'][1].append(pp0.sol[1])
                velocities['vel_out'][1].append(pp0.sol[0])
            else:
                '''
                w2 = (wave > wave_range[0]) & (wave < wave_range[1])

                pp0, _ = fit_and_clean(stars_templates, spectra[w2], velscale, start, goodpixels, lam,
                                       miles.lam_temp)
                newwave_out.append(np.copy(pp0.lam))
                flux_out.append(np.copy(pp0.galaxy))
                resid = pp0.galaxy - pp0.bestfit
                mask_out.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                newwave_out.append(np.copy(pp0.lam))
                flux_out.append(np.copy(pp0.galaxy))
                mask_out.append((abs(resid)<np.median(resid)+np.std(resid))*1)
                '''

                mask_out.append(np.zeros_like(wave[:-1]))
                resid_out.append(np.full_like(wave[:-1], np.nan))
                newwave_out.append(np.copy(wave[:-1]))
                flux_out.append(np.copy(spectra[:-1]))
                std_out.append(std)
                sres_out.append(np.copy(sres[:-1]))
                ivar_out.append(np.copy(1 / np.square(error[:-1])))
                velocities['vel_out'][1].append(np.nan)
                sigmas['sigma_out'][1].append(sigma)
            ifu_array.append(stacked[1].data[j][0])

            # '''
            # sigmas['sigma_out'][1].append(0)
        primary_hdu = fits.PrimaryHDU(np.zeros(2))
        wave = fits.ImageHDU(newwave_in[0], name='wave')
        # wave_fixed_in = fits.ImageHDU(np.array(newwave_in), name='wave_in')

        flux_fixed_in = fits.ImageHDU(np.array(flux_in), name='flux_in')
        flux_fixed_mid = fits.ImageHDU(np.array(flux_mid), name='flux_mid')
        flux_fixed_out = fits.ImageHDU(np.array(flux_out), name='flux_out')

        ivar_fixed_in = fits.ImageHDU(np.array(ivar_in), name='ivar_in')
        ivar_fixed_mid = fits.ImageHDU(np.array(ivar_mid), name='ivar_mid')
        ivar_fixed_out = fits.ImageHDU(np.array(ivar_out), name='ivar_out')

        sres_fixed_in = fits.ImageHDU(np.array(sres_in), name='sres_in')
        sres_fixed_mid = fits.ImageHDU(np.array(sres_mid), name='sres_mid')
        sres_fixed_out = fits.ImageHDU(np.array(sres_out), name='sres_out')

        mask_fixed_in = fits.ImageHDU(np.array(mask_in), name='mask_in')
        mask_fixed_mid = fits.ImageHDU(np.array(mask_mid), name='mask_mid')
        mask_fixed_out = fits.ImageHDU(np.array(mask_out), name='mask_out')

        resid_fixed_in = fits.ImageHDU(np.array(resid_in), name='resid_in')
        resid_fixed_mid = fits.ImageHDU(np.array(resid_mid), name='resid_mid')
        resid_fixed_out = fits.ImageHDU(np.array(resid_out), name='resid_out')

        sigma0_in = fits.Column(name='sigma0_in', array=sigmas['sigma_in'][0], format='E')
        sigma1_in = fits.Column(name='sigma1_in', array=sigmas['sigma_in'][1], format='E')

        sigma0_mid = fits.Column(name='sigma0_mid', array=sigmas['sigma_mid'][0], format='E')
        sigma1_mid = fits.Column(name='sigma1_mid', array=sigmas['sigma_mid'][1], format='E')

        sigma0_out = fits.Column(name='sigma0_out', array=sigmas['sigma_out'][0], format='E')
        sigma1_out = fits.Column(name='sigma1_out', array=sigmas['sigma_out'][1], format='E')

        error0_in = fits.Column(name='error_v_in', array=errors['error_in'][0], format='E')
        error1_in = fits.Column(name='error_s_in', array=errors['error_in'][1], format='E')

        error0_mid = fits.Column(name='error_v_mid', array=errors['error_mid'][0], format='E')
        error1_mid = fits.Column(name='error_s_mid', array=errors['error_mid'][1], format='E')

        error0_out = fits.Column(name='error_v_out', array=errors['error_out'][0], format='E')
        error1_out = fits.Column(name='error_s_out', array=errors['error_out'][1], format='E')

        vel0_in = fits.Column(name='vel0_in', array=velocities['vel_in'][0], format='E')
        vel1_in = fits.Column(name='vel1_in', array=velocities['vel_in'][1], format='E')

        vel0_mid = fits.Column(name='vel0_mid', array=velocities['vel_mid'][0], format='E')
        vel1_mid = fits.Column(name='vel1_mid', array=velocities['vel_mid'][1], format='E')

        vel0_out = fits.Column(name='vel0_out', array=velocities['vel_out'][0], format='E')
        vel1_out = fits.Column(name='vel1_out', array=velocities['vel_out'][1], format='E')

        sigma_hdu = fits.BinTableHDU.from_columns(
            [sigma0_in, sigma1_in, sigma0_mid, sigma1_mid, sigma0_out, sigma1_out,
             vel0_in, vel1_in, vel0_mid, vel1_mid, vel0_out, vel1_out,
             error0_in, error1_in, error0_mid, error1_mid, error0_out, error1_out], name='kinematics')

        plateifus = fits.Column(name='plateifu', array=np.array(ifu_array), format='32A')
        cols = fits.ColDefs([plateifus])
        pltifu = fits.BinTableHDU.from_columns(cols, name='info')
        try:
            np.save('std_in', np.array(std_in))
            np.save('std_mid', np.array(std_mid))
            np.save('std_out', np.array(std_out))
            std_fixed_in = fits.ImageHDU(np.array(std_in), name='std_in')
            std_fixed_mid = fits.ImageHDU(np.array(std_mid), name='std_mid')
            std_fixed_out = fits.ImageHDU(np.array(std_out), name='std_out')
            hdul = fits.HDUList([primary_hdu, pltifu, sigma_hdu, wave,
                                 flux_fixed_in, flux_fixed_mid, flux_fixed_out,
                                 mask_fixed_in, mask_fixed_mid, mask_fixed_out,
                                 resid_fixed_in, resid_fixed_mid, resid_fixed_out,
                                 ivar_fixed_in, ivar_fixed_mid, ivar_fixed_out,
                                 sres_fixed_in, sres_fixed_mid, sres_fixed_out,
                                 std_fixed_in, std_fixed_mid, std_fixed_out])
        except:
            hdul = fits.HDUList([primary_hdu, pltifu, sigma_hdu, wave,
                                 flux_fixed_in, flux_fixed_mid, flux_fixed_out,
                                 mask_fixed_in, mask_fixed_mid, mask_fixed_out,
                                 resid_fixed_in, resid_fixed_mid, resid_fixed_out,
                                 sres_fixed_in, sres_fixed_mid, sres_fixed_out,
                                 ivar_fixed_in, ivar_fixed_mid, ivar_fixed_out])

        hdul.writeto('stacked_sigfix.fits', overwrite=True)
        # hdul.writeto('stacked_mass_kpc_new_sigfix_' + str(i) + '.fits', overwrite=True)
        # if i_num == 0:
        #    hdul.writeto('stacked_sigfix_1mask.fits', overwrite=True)
        # else:
        #    hdul.writeto('stacked_sigfix_1wei.fits', overwrite=True)

        '''
        t = Table([sigmas['sigma_in'][0], sigmas['sigma_in'][1], sigmas['sigma_out'][0], sigmas['sigma_out'][0]],
                  names=['sigma_in0', 'sigma_in1', 'sigma_out0', 'sigma_out1'])

        t.write('sigmas_' + str(i) + '.fits', overwrite=True)
        '''
