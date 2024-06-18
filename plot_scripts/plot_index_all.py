import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import sys
import pyphot
from pyphot import unit
import nmmn.stats
import bces.bces as BCES
from ltsfit.lts_linefit import lts_linefit
import warnings
import multiprocessing
import pandas as pd

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    multiprocessing.freeze_support()
    colormap0 = mpl.cm.bwr

    stacked_path = '/Volumes/ZXYDisk/new_stack/'
    # stacked = fits.open(stacked_path + 'stacked_smooth_sigma300_mask.fits')
    # stacked=fits.open()
    split_num = int(sys.argv[1])
    split = fits.open('sample_split_outliers_total.fits')
    split_type = ['mtot_v_m20kpc', 'm10kpc_v_sigmacen', 'm10kpc_v_m20kpc_control', 'm20kpc_v_sigmacen', 'sigma_v_mtot']
    # path = '/Volumes/ZXYDisk/new_stack/'
    hdu = fits.open('index_all.fits')
    parameters = fits.open('parameters_spx2_max_reff2.fits')

    plateifus = hdu['info'].data['plateifu']
    ifus0, _, idx0 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type[split_num]] % 2 == 0],
                                    plateifus,
                                    return_indices=True)
    ifus1, _, idx1 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type[split_num]] % 2 == 1],
                                    plateifus,
                                    return_indices=True)
    lam1 = 2404
    # wave = stacked['WAVE'].data[432:lam1]
    # wave = wave / (
    #         1.0 + 0.05792105 / (238.0185 - (10000.0 / wave) ** 2) + 0.00167917 / (57.362 - (10000.0 / wave) ** 2))
    # l = pyphot.LickLibrary()

    radii = ['in', 'mid', 'out']
    val_med = {'Mg_b': [[0, 0, 0], [0, 0, 0]],
               'Fe5270': [[0, 0, 0], [0, 0, 0]],
               'Fe5335': [[0, 0, 0], [0, 0, 0]],
               'MgFe': [[0, 0, 0], [0, 0, 0]],
               'Mgb/Fe': [[0, 0, 0], [0, 0, 0]]}
    # for j in range(3):
    #     f = np.copy(stacked['flux_' + radii[j]].data[:, 432:lam1])
    #     fnor = np.zeros_like(f)
    #     nor = np.zeros_like(f)
    #
    #     for k in range(len(f)):
    #         idx_nan = np.isfinite(f[k])
    #         p0 = np.polyfit(wave[idx_nan], f[k][idx_nan], 10)
    #         p = np.poly1d(p0)
    #         nor[k] = p(wave)
    #         fnor[k] = f[k] / nor[k]
    #     mask = stacked['mask_' + radii[j]].data[:, 432:lam1]
    #     fnor[mask == 0] = np.nan
    #     f0 = np.nanmedian(fnor[idx0], axis=0)
    #     f1 = np.nanmedian(fnor[idx1], axis=0)
    #     fk = l['Mg_b']
    #     val_med['Mg_b'][0][j] = fk.get(wave * unit('AA'), f0, axis=1)
    #     val_med['Mg_b'][1][j] = fk.get(wave * unit('AA'), f1, axis=1)
    #
    #     fk = l['Fe5270']
    #     val_med['Fe5270'][0][j] = fk.get(wave * unit('AA'), f0, axis=1)
    #     val_med['Fe5270'][1][j] = fk.get(wave * unit('AA'), f1, axis=1)
    #
    #     fk = l['Fe5335']
    #     val_med['Fe5335'][0][j] = fk.get(wave * unit('AA'), f0, axis=1)
    #     val_med['Fe5335'][1][j] = fk.get(wave * unit('AA'), f1, axis=1)
    #
    #     val_med['MgFe'][0][j] = np.sqrt(val_med['Mg_b'][0][j] * (0.72 * val_med['Fe5270'][0][j] + 0.28 * val_med['Fe5335'][0][j]))
    #     val_med['MgFe'][1][j] = np.sqrt(val_med['Mg_b'][1][j] * (0.72 * val_med['Fe5270'][1][j] + 0.28 * val_med['Fe5335'][1][j]))
    #
    #     val_med['Mgb/Fe'][0][j] = val_med['Mg_b'][0][j] / (0.5 * (val_med['Fe5270'][0][j] + val_med['Fe5335'][0][j]))
    #     val_med['Mgb/Fe'][1][j] = val_med['Mg_b'][1][j] / (0.5 * (val_med['Fe5270'][1][j] + val_med['Fe5335'][1][j]))

    _, _, idp = np.intersect1d(plateifus, parameters[1].data['plateifu'], return_indices=True)
    _, _, idp0 = np.intersect1d(ifus0, parameters[1].data['plateifu'], return_indices=True)
    _, _, idp1 = np.intersect1d(ifus1, parameters[1].data['plateifu'], return_indices=True)
    radii2 = ['cen', 'mid', 'out']
    names_cal = {'Mg_b': r'$\mathbf{Mg_b}$', 'Fe5270': r'$\mathbf{Fe5270}$',  # 'Fe5335': r'$\mathbf{Fe5335}$',
                 'MgFe': r'$\mathbf{[MgFe]}$', 'Mgb/Fe': r'$\mathbf{Mg_b/<Fe>}$'}
    names_p = ['mgb', 'fe5270', 'mgfe', 'mgbfe']
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Charter",
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "axes.labelsize": 21,
        "axes.labelweight": "bold"
    })
    fig, ax = plt.subplots(4, 3, figsize=(12, 16), clear=True)
    split = fits.open('sample_split_outliers_total.fits')
    split_type = ['mtot_v_m20kpc', 'm10kpc_v_sigmacen', 'm10kpc_v_m20kpc_control', 'm20kpc_v_sigmacen', 'sigma_v_mtot']
    split_num = int(sys.argv[1])
    normalize = mcolors.Normalize(vmin=-1.3, vmax=0.3)
    colormap = mpl.cm.PRGn
    color_low = colormap(normalize(-1))
    color_high = colormap(normalize(0))
    colors = [color_high, color_low]
    edgecolors = ['xkcd:forest', 'xkcd:blue violet']
    shapes = ['o', 'd']
    labels = [r'$\mathrm{Extended}$', r'$\mathrm{Compact}$']
    if (split_num == 3) | (split_num == 1):
        colors = ['xkcd:tangerine', 'xkcd:dark sky blue']
        edgecolors = ['xkcd:pumpkin', 'xkcd:vivid blue']
        shapes = ['p', 's']
        labels = [r'$\mathrm{Higher\ \sigma}$', r'$\mathrm{Lower\ \sigma}$']
    ylims = {'Mg_b': [1.5, 6], 'Fe5270': [0, 4], 'Fe5335': [0, 4], 'MgFe': [1, 3.5], 'Mgb/Fe': [0, 4]}
    # ylims = {'Mgb': [0, 6], 'Fe5270': [0, 7], 'Fe5335': [0, 4], 'MgFe': [1, 6], 'MgbFe': [0, 2.7]}
    __spec__ = None
    xdata = np.log10(hdu['sigma'].data[0])
    errx = parameters[1].data['sigma_' + radii2[0] + '_noise'][idp] / (
            parameters[1].data['sigma_' + radii2[0]][idp] * np.log(10))
    dataf={'sigcen':xdata,'sigcen_err':errx}
    scatter_lira={'Mg_b': [[0.285,0.479,0.8773],[0.00786,0.0133,0.0244]],
                  'Fe5270': [[0.190,0.4103,0.6995],[0.00578,0.0115,0.021]],  # 'Fe5335': r'$\mathbf{Fe5335}$',
                  'MgFe': [[0.171,0.289,0.4743],[0.0048,0.008,0.0141]],
                  'Mgb/Fe': [[0.1887,0.5144,1.386],[0.005,0.01395,0.0382]]}

    for j, name in enumerate([*names_cal]):
        ax[j][1].set_title(names_cal[name], fontsize=20)
        for i in range(3):
            if j < 3:
                ax[j][i].xaxis.set_visible(False)
            if i > 0:
                ax[j][i].yaxis.set_visible(False)
            xdata = np.log10(hdu['sigma'].data[0])
            errx = parameters[1].data['sigma_' + radii2[0] + '_noise'][idp] / (
                    parameters[1].data['sigma_' + radii2[0]][idp] * np.log(10))

            # ax[j][i].set_xlim(ylims[name])
            # data_tmp = hdu[name].data[i]
            # bins = np.histogram_bin_edges(data_tmp[(data_tmp > ylims[name][0]) & (data_tmp < ylims[name][1])], bins='scott')
            # counts_tmp, _ = np.histogram(hdu[name].data[i], bins=bins)
            # ax[j][i].bar(bins[1:], counts_tmp / np.nansum(counts_tmp), width=bins[1] - bins[0],
            #              color='grey', align='center', alpha=0.5)
            # counts_tmp, _ = np.histogram(hdu[name].data[i][idx0], bins=bins)
            # ax[j][i].step(bins[1:], counts_tmp / np.nansum(counts_tmp),
            #               color=colors[0], where='mid', linewidth=2)
            # counts_tmp, _ = np.histogram(hdu[name].data[i+1][idx1], bins=bins)
            # ax[j][i].step(bins[1:], counts_tmp / np.nansum(counts_tmp),
            #               color=colors[1], where='mid', linewidth=2)

            ax[j][i].set_ylim(ylims[name])
            # ax[j][i].scatter(np.log10(hdu['sigma'].data[0]),
            #                  hdu[name].data[i], s=10, marker='.', facecolor='grey', edgecolor='none')

            # ax[j][i].scatter(np.log10(hdu['sigma'].data[i][idx0]),
            #                  hdu[name].data[i][idx0], marker=shapes[0], c=colors[0],
            #                  edgecolor=edgecolors[0])
            # ax[j][i].scatter(np.log10(hdu['sigma'].data[i][idx1]),
            #                  hdu[name].data[i][idx1], marker=shapes[1], c=colors[1],
            #                  edgecolor=edgecolors[1])
            nboot = 10000

            # xdata=parameters[1].data['massout(max-20kpc)'][idp]
            # errx=xdata*0.001
            ydata = hdu[name].data[i]
            erry = hdu[name + '_err'].data[i]
            dataf[name+radii[i]]=ydata
            dataf[name+radii[i]+'err']=erry

            mask_y = (ydata > ylims[name][0]) & (~np.isnan(erry)) & (ydata < ylims[name][1]) & (~np.isnan(xdata))
            xdata = np.array(xdata[mask_y], dtype=float)
            errx = np.array(errx[mask_y], dtype=float)
            ydata = np.array(ydata[mask_y], dtype=float)
            erry = np.array(erry[mask_y], dtype=float)
            cov = np.array(np.zeros_like(xdata), dtype=float)
            # normalize0 = mcolors.Normalize(vmin=2, vmax=2.7)
            # c=np.log10(hdu['sigma'].data[0][mask_y])

            normalize0 = mcolors.Normalize(vmin=10, vmax=11.5)
            c = parameters[1].data['massout(max-20kpc)'][idp]
            c = c[mask_y]

            ax[j][i].scatter(xdata, ydata,
                             s=10, marker='o', color='grey')

            ax[j][i].errorbar(xdata, ydata, xerr=errx, yerr=erry,
                              ecolor='grey', ls='none', linewidth=0.3, capsize=0, capthick=0.5, alpha=0.6)
            x = np.linspace(xdata.min(), xdata.max())


            def func(x):
                return x[1] * x[0] + x[2]


            # try:
            #     p = lts_linefit(xdata, ydata, errx, erry, plot=False)
            #     ax[j][i].scatter(xdata[~p.mask], ydata[~p.mask], facecolor='None', edgecolor='red', lw=0.5)
            #     ax[j][i].plot(x, p.ab[1] * x + p.ab[0], '--k', label="LTS $y|x$")
            #     ax[j][i].text(0.05, 0.17,
            #                   '$\mathrm{\sigma_e^{LTS}=' + format(p.sig_int, '.4f') + '^{\pm' + format(p.sig_int_err,
            #                                                                                            '.4f') + '}}$',
            #                   transform=ax[j][i].transAxes, fontsize=20)
            #
            # except:
            #     print('ouch')
            # ax[j][i].plot(xdata,p.coef)

            a, b, erra, errb, covab = BCES.bcesp(xdata, errx, ydata, erry, cov, nboot)
            bcesMethod = 0
            fitm = np.array([a[bcesMethod], b[bcesMethod]])
            covm = np.array([(erra[bcesMethod] ** 2, covab[bcesMethod]), (covab[bcesMethod], errb[bcesMethod] ** 2)])
            lcb, ucb, xcb = nmmn.stats.confbandnl(xdata, ydata, func, fitm, covm, 2, 0.954, x)
            # break
            ax[j][i].plot(xcb, a[bcesMethod] * xcb + b[bcesMethod], '-k', label="BCES $y|x$")
            residual = a[bcesMethod] * xdata + b[bcesMethod] - ydata
            scatter_in = np.nanstd(a[bcesMethod] * xdata + b[
                bcesMethod] - ydata)  # np.nansum((residual-np.nanmean(residual))**2)/len(xdata)#np.nanstd(a[bcesMethod] * xdata + b[bcesMethod]-ydata)
            ax[j][i].text(0.05, 0.04, '$\mathrm{\sigma_e^{BCES}=' + format(scatter_in, '.4f') + '}$',
                          transform=ax[j][i].transAxes, fontsize=20)
            ax[j][i].text(0.05, 0.17, '$\mathrm{\sigma_e^{lira}=' + format(scatter_lira[name][0][i], '.4f') +
                          '^{\pm' + format(scatter_lira[name][1][i], '.4f')+ '}}$',
                          transform=ax[j][i].transAxes, fontsize=20)
            ax[j][i].fill_between(xcb, lcb, ucb, alpha=0.5, facecolor='orange')
            sign = ['-', '+'][b[bcesMethod] > 0]
            # ax[j][i].plot(x, lcb, 'k--')
            # ax[j][i].plot(x, ucb, 'k--')
            ax[j][i].text(0.05, 0.9, '$\mathrm{' + format(a[bcesMethod], '.4f') +
                          '^{\pm' + format(erra[bcesMethod], '.4f') + '} \cdot log\ \sigma}$',
                          transform=ax[j][i].transAxes, fontsize=20)

            ax[j][i].text(0.07, 0.75,
                          '$\mathrm{' + sign + format(abs(b[bcesMethod]), '.4f') + '^{\pm' + format(errb[bcesMethod],
                                                                                                    '.4f') + '}}$',
                          transform=ax[j][i].transAxes, fontsize=20)

            # ax[j][i].set_xlim([np.nanmin(np.log10(hdu['sigma'].data[0])), np.nanmax(np.log10(hdu['sigma'].data[0]))])

            # -------------------------------------
            # --------manga measurement------------
            # -------------------------------------

            # ------------hist---------------------
            # data_tmp = parameters[1].data[name + '_' + radii2[i]][idp]
            # bins = np.histogram_bin_edges(data_tmp[(data_tmp > ylims[name][0]) & (data_tmp < ylims[name][1])], bins='scott')
            # counts_tmp, _ = np.histogram(parameters[1].data[name + '_' + radii2[i]][idp], bins=bins)
            # ax[j][i].bar(bins[1:], counts_tmp / np.nansum(counts_tmp), width=bins[1] - bins[0],
            #              color='grey', align='center', alpha=0.5)
            # counts_tmp, _ = np.histogram(parameters[1].data[name + '_' + radii2[i]][idp0], bins=bins)
            # ax[j][i].step(bins[1:], counts_tmp / np.nansum(counts_tmp),
            #               color=colors[0], where='mid', linewidth=2)
            # counts_tmp, _ = np.histogram(parameters[1].data[name + '_' + radii2[i]][idp1], bins=bins)
            # ax[j][i].step(bins[1:], counts_tmp / np.nansum(counts_tmp),
            #               color=colors[1], where='mid', linewidth=2)
            # ax[j][i].plot(np.full(50,val_med[name][0][i]),np.linspace(0,0.2),color=colors[0])
            # ax[j][i].plot(np.full(50,val_med[name][1][i]),np.linspace(0,0.2),color=colors[1])

            # ------------scatter---------------------
            # ax[j][i].scatter(np.log10(parameters[1].data['sigma_' + radii2[i]][idp]),
            #                  parameters[1].data[name + '_' + radii2[i]][idp], facecolor='grey', edgecolor='none')
            # ax[j][i].scatter(np.log10(parameters[1].data['sigma_' + radii2[i]][idp0]),
            #                  parameters[1].data[name + '_' + radii2[i]][idp0], marker=shapes[0], c=colors[0],
            #                  edgecolor=edgecolors[0])
            # ax[j][i].scatter(np.log10(parameters[1].data['sigma_' + radii2[i]][idp1]),
            #                  parameters[1].data[name + '_' + radii2[i]][idp1], marker=shapes[1], c=colors[1],
            #                  edgecolor=edgecolors[1])
    ax[-1][1].set_xlabel(r'$\mathrm{log_{10}\ \sigma(R<0.5R_e)/(km/s)}$', fontsize=20)
    # ax[-1][1].set_xlabel(r'$\mathrm{log_{10}\ M_*(R>20kpc)/(M_\odot)}$', fontsize=20)

    ax[-1][0].legend()

    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    fig.savefig('index_all_' + split_type[split_num] + '_sigmain.pdf', dpi=300, bbox_inches='tight')
    # mask=~np.isnan(dataf['sigcen'])
    # for name in [*dataf]:
    #     mask=mask&(~np.isnan(dataf[name]))
    # for name in [*dataf]:
    #    dataf[name]=dataf[name][mask]
    #
    #
    # df = pd.DataFrame(dataf)
    # df.to_csv('index_all.csv')
    #
