import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import sys
import astropy.cosmology as cosmology
from matplotlib.legend_handler import HandlerTuple
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    split_num = 3#int(sys.argv[1])
    titles = ['', '', r'$\mathbf{Compact\ vs.\ Extended}$', r'$\mathbf{Low\ \sigma\ vs.\ High\ \sigma}$']

    parameters = fits.open('parameters_spx2_max_reff2.fits')
    manga_sga = fits.open('manga_sga_z.fits')
    sga_mass = fits.open('sga_mass_new.fits')
    mask_hdu = fits.open('sample_mask.fits')
    split = fits.open('sample_split_outliers_total.fits')
    split_grd=fits.open('sample_split_gradients.fits')
    names = ['Mg_b', 'Fe5270', 'MgFe', 'Mgb/Fe']
    names_path =['mgb','fe5270','mgfe','mgbfe']

    sga = fits.open('/Volumes/ZXYDisk/new_stack/SGA-2020.fits')
    '/Volumes/ZXYDisk/new_stack/SGA-2020.fits'
    stack = fits.open('/Volumes/ZXYDisk/new_stack/stacked_smooth_sigma300_mask.fits')
    cosm = cosmology.FlatLambdaCDM(70, 0.3, 2.725, Ob0=0.046)

    morph_mask = (parameters[1].data['gzoo'][:, 0] != 1) | (parameters[1].data['ttype'] <= 0)
    gals = stack['info'].data['plateifu']
    _, _, idx_morph = np.intersect1d(parameters[1].data['plateifu'][morph_mask], gals, return_indices=True)
    gals = gals[idx_morph]

    size = len(gals)
    sga_m = sga_mass[1].data['mass_sga_rc15']
    sga_ifu = sga_mass[1].data['plateifu']
    _, idx, _ = np.intersect1d(sga_ifu, gals, return_indices=True)
    split_type = ['mtot_v_m20kpc', 'm10kpc_v_sigmacen', 'm10kpc_v_m20kpc_control', 'm20kpc_v_sigmacen', 'sigma_v_mtot']

    total_mass = sga_m[idx]
    _, idx_params, _ = np.intersect1d(parameters[1].data['plateifu'], gals, return_indices=True)
    sigma_cen = np.log10(parameters[1].data['sigma_cen'][idx_params])
    m_in_10kpc = parameters[1].data['mass10kpc'][idx_params]
    m_out_20kpc = parameters[1].data['massout(max-20kpc)'][idx_params]
    select_data = {'mtot_v_m20kpc': [total_mass, m_out_20kpc,
                                     r'$\mathrm{log_{10}\ M_*/M_\odot}$',
                                     r'$\mathrm{log_{10}\ M_*(R>20kpc)/M_\odot}$',
                                     [11.1, 12.12], [9.75, 12.25]],
                   'm10kpc_v_sigmacen': [m_in_10kpc, sigma_cen,
                                         r'$\mathrm{log_{10}\ M_*(R<10kpc)/M_\odot}$',
                                         r'$\mathrm{log_{10}\ \sigma(R<0.5R_e)/(km/s)}$',
                                         [11.0, 11.85], [2.0, 2.68]],
                   'm10kpc_v_m20kpc_control': [m_in_10kpc, m_out_20kpc,
                                               r'$\mathrm{log_{10}\ M_*(R<10kpc)/M_\odot}$',
                                               r'$\mathrm{log_{10}\ M_*(R>20kpc)/M_\odot}$',
                                               [11.0, 11.85], [9.75, 12.25]],
                   'm20kpc_v_sigmacen': [m_out_20kpc, sigma_cen,
                                         r'$\mathrm{log_{10}\ M_*(R>20kpc)/M_\odot}$',
                                         r'$\mathrm{log_{10}\ \sigma(R<0.5R_e)/(km/s)}$',
                                         [9.75, 12.25], [2.0, 2.68]],
                   'sigma_v_mtot': [sigma_cen, total_mass,
                                    r'$\mathrm{log_{10}\ \sigma(R<0.5R_e)/(km/s)}$',
                                    r'$\mathrm{log_{10}\ M_*/M_\odot}$',
                                    [2.0, 2.68], [11.1, 12.22]]}
    dx = select_data[split_type[split_num]][0]
    dy = select_data[split_type[split_num]][1]

    plateifuse = []
    _, idx, _ = np.intersect1d(mask_hdu[1].data['plateifu'], gals, return_indices=True)
    _, idxx_m, idx_m = np.intersect1d(gals, sga_mass[1].data['plateifu'], return_indices=True)

    line_type = ['-', '--']
    # labels_markers = ['High', 'Low']
    plt.clf()
    normalize = mcolors.Normalize(vmin=-1.3, vmax=0.3)
    colormap = mpl.cm.PRGn
    color_low = colormap(normalize(-1))
    color_high = colormap(normalize(0))
    colors = [color_high, color_low]
    shapes = ['o', 'd']
    edgecolors = ['xkcd:forest', 'xkcd:blue violet']
    labels_markers = [r'$\mathrm{Extended}$', r'$\mathrm{Compact}$']

    if (split_num == 3) | (split_num == 1):
        colors = ['xkcd:tangerine', 'xkcd:dark sky blue']
        edgecolors = ['xkcd:pumpkin', 'xkcd:vivid blue']
        labels_markers = [r'$\mathrm{Higher\ \sigma}$', r'$\mathrm{Lower\ \sigma}$']

        shapes = ['p', 's']

    fig, ax = plt.subplots(2, 3, figsize=(13, 8), clear=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Charter",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 16,
        "axes.labelweight": "bold"
        })
    _, idxx_p, idx_p = np.intersect1d(split[1].data['plateifu'], gals,
                                      return_indices=True)
    ifus0, idxx_p0, idx_p0 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type[split_num]] % 2 == 0],
                                            gals,
                                            return_indices=True)
    ifus1, idxx_p1, idx_p1 = np.intersect1d(split[1].data['plateifu'][split[1].data[split_type[split_num]] % 2 == 1],
                                            gals,
                                            return_indices=True)
    ax[0][0].scatter(dx, dy, facecolor='grey', edgecolor='none')
    ax[0][0].scatter(dx[idx_p0], dy[idx_p0], marker=shapes[0], c=colors[0], edgecolor=edgecolors[0])
    ax[0][0].scatter(dx[idx_p1], dy[idx_p1], marker=shapes[1], c=colors[1], edgecolor=edgecolors[1])

    ax[0][0].set_xlabel(select_data[split_type[split_num]][2], labelpad=2,fontsize=18)
    ax[0][0].set_ylabel(select_data[split_type[split_num]][3],fontsize=18)
    ax[0][0].set_xlim(select_data[split_type[split_num]][4])
    ax[0][0].set_ylim(select_data[split_type[split_num]][5])

    # ----------- Redshift -----------
    _, idxx_z, idx_z = np.intersect1d(gals, manga_sga[1].data['plateifu'], return_indices=True)
    bins_z = np.histogram_bin_edges(manga_sga[1].data['redshift'][idx_z], bins='scott')
    counts_tmp, _ = np.histogram(manga_sga[1].data['redshift'][idx_z], bins=bins_z)
    ax[1][2].bar(bins_z[1:], counts_tmp / np.nansum(counts_tmp), width=bins_z[1] - bins_z[0], color='grey',
                 label=r'$\mathrm{Total}$', align='center', alpha=0.5)

    _, _, idx_z0 = np.intersect1d(ifus0, manga_sga[1].data['plateifu'], return_indices=True)

    counts_tmp, _ = np.histogram(manga_sga[1].data['redshift'][idx_z0],
                                 bins=bins_z)
    ax[1][2].step(bins_z[1:], counts_tmp / np.nansum(counts_tmp), color=colors[0],
                  label=labels_markers[0], where='mid', linewidth=2)

    _, _, idx_z1 = np.intersect1d(ifus1, manga_sga[1].data['plateifu'], return_indices=True)
    counts_tmp, _ = np.histogram(manga_sga[1].data['redshift'][idx_z1],
                                 bins=bins_z)
    ax[1][2].step(bins_z[1:], counts_tmp / np.nansum(counts_tmp), color=colors[1],
                  label=labels_markers[1], where='mid', linewidth=2, ls='--')
    ax[1][2].set_xlabel(r'$\mathrm{Redshift}$', labelpad=2,fontsize=18)

    # ----------- Total Mass -----------
    bins_m = np.histogram_bin_edges(sga_mass[1].data['mass_sga_rc15'][idx_m], bins='scott')
    counts_tmp, _ = np.histogram(sga_mass[1].data['mass_sga_rc15'][idx_m], bins=bins_m)
    ax[1][0].bar(bins_m[1:], counts_tmp / np.nansum(counts_tmp), width=bins_m[1] - bins_m[0],
                 color='grey', label=r'$\mathrm{Total}$', align='center', alpha=0.5)

    _, _, idx_m0 = np.intersect1d(ifus0, sga_mass[1].data['plateifu'], return_indices=True)
    counts_tmp, _ = np.histogram(sga_mass[1].data['mass_sga_rc15'][idx_m0], bins=bins_m)
    ax[1][0].step(bins_m[1:], counts_tmp / np.nansum(counts_tmp),
                  color=colors[0], label=labels_markers[0], where='mid', linewidth=2)
    ax[1][0].set_ylabel(r'$\mathrm{Fraction}$', labelpad=2,fontsize=18)

    _, _, idx_m1 = np.intersect1d(ifus1, sga_mass[1].data['plateifu'], return_indices=True)
    counts_tmp, _ = np.histogram(sga_mass[1].data['mass_sga_rc15'][idx_m1], bins=bins_m)
    ax[1][0].step(bins_m[1:], counts_tmp / np.nansum(counts_tmp),
                  color=colors[1], label=labels_markers[1], where='mid', linewidth=2, ls='--')

    ax[1][0].set_xlabel(r'$\mathrm{log_{10}\ M_*/M_\odot}$', labelpad=2,fontsize=18)

    # ----------- velocity dispersion -----------
    data_tmp = np.log10(parameters[1].data['sigma_cen'][idx_params])
    bins_p = np.histogram_bin_edges(data_tmp[~np.isnan(data_tmp)], bins='scott')
    counts_tmp, _ = np.histogram(data_tmp, bins=bins_p)
    ax[1][1].bar(bins_p[1:], counts_tmp / np.nansum(counts_tmp), width=bins_p[1] - bins_p[0],
                 color='grey', alpha=0.5, label=r'$\mathrm{Total}$', align='center')

    _, _, idx_tmp0 = np.intersect1d(ifus0, parameters[1].data['plateifu'], return_indices=True)
    counts_tmp, _ = np.histogram(np.log10(parameters[1].data['sigma_cen'][idx_tmp0]), bins=bins_p)
    ax[1][1].step(bins_p[1:], counts_tmp / np.nansum(counts_tmp),
                  color=colors[0], label=labels_markers[0], where='mid', linewidth=2)

    _, _, idx_tmp1 = np.intersect1d(ifus1, parameters[1].data['plateifu'], return_indices=True)
    counts_tmp, _ = np.histogram(np.log10(parameters[1].data['sigma_cen'][idx_tmp1]), bins=bins_p)
    ax[1][1].step(bins_p[1:], counts_tmp / np.nansum(counts_tmp),
                  color=colors[1], label=labels_markers[1], where='mid', linewidth=2, ls='--')
    # ----------------
    mydx = np.array([])
    myflux = np.empty((2, 100, 0)).tolist()
    file_path = '/Volumes/ZXYDisk/sga_ellipse/'
    mask0 = split[1].data[split_type[split_num]] >= 0
    split_color = split[1].data[split_type[split_num]][mask0]
    plateifus = split[1].data['plateifu'][mask0]
    _, idx_split, idx = np.intersect1d(plateifus, manga_sga[1].data['plateifu'], return_indices=True)

    for i, gal in enumerate(manga_sga[1].data[idx]):
        tp_tmp = int(split_color[idx_split[i]])
        sid = gal['sgaid']

        idx_sga = np.where(sga[1].data['sga_id'] == sid)[0][0]
        gn = sga[1].data['group_name'][idx_sga]
        ra = gal['objra']
        dec = gal['objdec']
        z_sga = sga[1].data['z_leda'][idx_sga]
        d_a = cosm.angular_diameter_distance(z_sga).to('kpc').value
        d_l = cosm.luminosity_distance(z_sga).to('pc').value

        hdu = fits.open(file_path + gn + '-largegalaxy-' + str(sid) + '-ellipse.fits')

        dx = hdu[1].data['R_SMA'][0] * 0.263 * (d_a * np.pi / (3600 * 180))
        mydx = np.insert(mydx, 0, dx)
        idx_10 = np.where(abs(dx - 10) == np.nanmin(abs(dx - 10)))[0][0]
        #flux= 22.5 - 2.5 * np.log10(hdu[1].data['R_INTENS'][0]) - 5 * np.log10(d_l / 10)
        flux=np.log10(hdu[1].data['R_INTENS'][0]/hdu[1].data['R_INTENS'][0][idx_10])
        #flux = np.log10(hdu[1].data['R_INTENS'][0] / np.nanmedian(hdu[1].data['R_INTENS'][0][(dx < 30) & (dx > 5)]))

        for j in range(100):
            myflux[tp_tmp][j].append(np.nanmedian(flux[(dx >= j) & (dx <= j + 1)]))
        ax[0][1].plot(dx ** 0.25, flux, color=colors[tp_tmp], alpha=0.2)
    ax[0][1].set_xlim(left=1.)
    ylims = [-2.5, 2]
    #ylims=[-20,-8]
    ax[0][1].set_ylim(ylims)
    ax[0][1].plot(np.full(50, 10 ** 0.25), np.linspace(ylims[0], ylims[1]), 'k-')
    ax[0][1].plot(np.full(50, 20 ** 0.25), np.linspace(ylims[0], ylims[1]), 'k:')
    mymedian = []
    if split_num==3:
        mediancolors=['xkcd:chestnut', 'xkcd:vivid blue']
    else:
        mediancolors=edgecolors
    ax[0][1].plot(np.linspace(0, 100, num=100) ** 0.25, [np.nanmedian(myflux[0][k]) for k in range(100)],
                  color=mediancolors[0],
                  lw=2)
    ax[0][1].plot(np.linspace(0, 100, num=100) ** 0.25, [np.nanmedian(myflux[1][k]) for k in range(100)],
                  color=mediancolors[1],
                  lw=2)
    ax[0][1].text(10 ** 0.25-0.1, -0.3,'10kpc',ha='right',fontsize=15)
    ax[0][1].text(20 ** 0.25+0.1, 0,'20kpc',ha='left',fontsize=15)

    #ax[0][1].legend(fontsize=14, bbox_to_anchor=(0.46, 0.6), frameon=False)

    ax[1][1].set_xlabel(r'$\mathrm{log_{10}\ \sigma(km/s)}$', fontname="Times New Roman", labelpad=2,fontsize=18)
    ax[0][1].set_title(titles[split_num], fontsize=21)
    ax[0][-1].axis('off')
    t1 = ax[0][-1].scatter([], [], s=300,marker='s',facecolor='grey', edgecolor='None',
                        label=r'$\mathrm{Total}$', alpha=0.5)
    t2 = ax[0][-1].scatter([], [], s=100, facecolor='grey', edgecolor='none')
    h2 = ax[0][-1].scatter([], [], s=100, c=colors[0],
                            marker=shapes[0], edgecolor=edgecolors[0], label=labels_markers[0])
    h1, = ax[0][-1].plot([], [], color=colors[0])
    l2 = ax[0][-1].scatter([], [], s=100, c=colors[1],
                            marker=shapes[1], edgecolor=edgecolors[1], label=labels_markers[1])
    l1, = ax[0][-1].plot([], [],'--', color=colors[1])
    #ax[0][1].tick_params(bottom=True, top=False, left=False, right=True)
    #ax[0][1].yaxis.set_label_position('right')
    ax[0][1].set_ylabel(r'$\mathrm{log_{10}\ F/F(10kpc)}$',fontsize=18, labelpad=2)
    ax[0][1].yaxis.set_label_coords(-0.12, 0.5, transform=ax[0][1].transAxes)
    ax[0][1].set_xlabel(r'$\mathrm{R^{1/4}(kpc)}$',fontsize=18)


    handles = [(t1, t2), (h1, h2), (l1, l2)]
    _, labels = ax[0][-1].get_legend_handles_labels()

    ax[0][-1].legend(handles=handles, labels=labels, fontsize=20,
                      handler_map={tuple: HandlerTuple(None)},loc='center')

    fig.subplots_adjust(wspace=0.25, hspace=0.3)
    plt.close('all')


