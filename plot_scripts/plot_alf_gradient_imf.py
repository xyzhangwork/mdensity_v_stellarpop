import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import sys
from read_alf import Alf
from matplotlib.legend_handler import HandlerTuple

# split_num = int(sys.argv[1])

comp_path = '../data/alf/full/'
path = '../data/alf/simple/'

split = fits.open('../data/sample_split_outliers_total.fits')
r_median = fits.open('../data/rmax_median_re2.fits')
rnames = ['rin_median(kpc)', 'rmid_median(kpc)', 'rout_median(kpc)']

split_type = ['mtot_v_m10kpc', 'sigma_v_mtot', 'm20kpc_v_sigmacen', 'm10kpc_v_m20kpc_control']
label_names = ['high', 'low']
radius_name = ['in', 'mid', 'out']
labels = ['High', 'Low']

normalize = mcolors.Normalize(vmin=-1.3, vmax=0.3)
colormap = mpl.cm.PRGn
color_low = colormap(normalize(-1))
color_high = colormap(normalize(0))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Charter",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 20,
    "axes.labelweight": "bold"
})
fig, ax = plt.subplots(2, 3, figsize=(16, 10), clear=True)

for split_num in [2, 3]:
    colors = [color_high, color_low]
    edgecolors = ['xkcd:forest', 'xkcd:blue violet']
    shapes = ['o', 'd']
    sample_labels = [r'$\mathrm{Extended}$', r'$\mathrm{Compact}$']
    xmin, xmax = 10, 10

    if split_num == 2:
        colors = ['xkcd:tangerine', 'xkcd:dark sky blue']
        edgecolors = ['xkcd:pumpkin', 'xkcd:vivid blue']
        sample_labels = [r'$\mathrm{Higher\ \sigma}$', r'$\mathrm{Lower\ \sigma}$']
        shapes = ['p', 's']

    ele_abundances = ['Mg']
    params = ['Mg']
    keywords = np.array(['Mg/Fe', 'IMF1', 'IMF2'])
    titles_full = [r'$\mathbf{[Mg/Fe]}$', r'$\texttt{imf1}$', r'$\texttt{imf2}$']

    data = {}
    err = {}

    for i in range(2):
        for name in keywords:
            data[name] = [[[], [], []], [[], [], []]]
            err[name] = [[[], [], []], [[], [], []]]

    for i in range(2):
        for j in range(3):
            myalf = Alf(
                path + split_type[split_num] + '/' + label_names[i] + '_' + split_type[split_num] + '_median_'
                + radius_name[j])
            myalf.get_total_met()
            myalf.abundance_correct()

            for name in ele_abundances:
                data[name + '/Fe'][i][j].append(myalf.xFe[name]['cl50'])
                err[name + '/Fe'][i][j].append(myalf.xFe[name]['std'])
            data['IMF1'][i][j].append(myalf.results['IMF1'][5])
            err['IMF1'][i][j].append(myalf.results['IMF1'][2])
            data['IMF2'][i][j].append(myalf.results['IMF2'][5])
            err['IMF2'][i][j].append(myalf.results['IMF2'][2])

    comp_data = {}
    comp_err = {}
    for i in range(2):
        for name in keywords:
            comp_data[name] = [[[], [], []], [[], [], []]]
            comp_err[name] = [[[], [], []], [[], [], []]]

    for i in range(2):
        for j in range(3):
            comp_alf = Alf(
                comp_path + split_type[split_num] + '/' + label_names[i] + '_' + split_type[split_num] +
                ['_rm_', '_red_median_'][split_num == 2]
                + radius_name[j])
            comp_alf.get_total_met()
            comp_alf.abundance_correct()
            comp_data['IMF1'][i][j].append(comp_alf.results['IMF1'][5])
            comp_err['IMF1'][i][j].append(comp_alf.results['IMF1'][2])
            comp_data['IMF2'][i][j].append(comp_alf.results['IMF2'][5])
            comp_err['IMF2'][i][j].append(comp_alf.results['IMF2'][2])

            for name in ele_abundances:
                comp_data[name + '/Fe'][i][j].append(comp_alf.xFe[name]['cl50'])
                comp_err[name + '/Fe'][i][j].append(comp_alf.xFe[name]['std'])

    for idx_cal, name in enumerate(keywords):
        ax[1][idx_cal].set_xlabel(r'$R\mathrm{/kpc}$', fontsize=21)
        for i in range(2):
            mask = split[1].data[split_type[split_num]] % 2 == i
            _, idx, _ = np.intersect1d(r_median[1].data['plateifu'], split[1].data['plateifu'][mask],
                                       return_indices=True)
            for j in range(3):
                # r_tmp=0.5 * j + i * 0.04
                r_tmp = np.nanmedian(r_median[1].data[rnames[j]][idx])
                if r_tmp < xmin:
                    xmin = np.copy(r_tmp)
                if r_tmp > xmax:
                    xmax = np.copy(r_tmp)

                ax[split_num - 2][idx_cal].errorbar([r_tmp], [comp_data[name][i][j]], yerr=[comp_err[name][i][j]],
                                                    ecolor=colors[i], linewidth=3, capsize=10, capthick=4,
                                                    linestyle=None,
                                                    alpha=1)

                ax[split_num - 2][idx_cal].scatter([r_tmp], [comp_data[name][i][j]], s=200,
                                                   edgecolors=edgecolors[i], facecolors=colors[i], linewidths=3,
                                                   marker=shapes[i],
                                                   alpha=1)
                if idx_cal == 0:
                    ax[split_num - 2][idx_cal].scatter([r_tmp], [data[name][i][j]], s=200,
                                                       edgecolors=edgecolors[i], facecolors='None', linewidths=3,
                                                       marker=shapes[i],
                                                       alpha=1)
                if split_num == 2:
                    ax[split_num - 2][idx_cal].set_title(titles_full[idx_cal], x=.5, y=1.04, fontsize=23)
    l1, = ax[split_num - 2][1].plot(np.linspace(xmin, xmax), np.full(50, 1.3), 'k--', lw=1.5)
    ax[split_num - 2][2].plot(np.linspace(xmin, xmax), np.full(50, 2.3), 'k--', lw=1.5)

    s1 = ax[split_num - 2][1].scatter([], [], s=200, marker=shapes[0], facecolor=colors[0], edgecolor=edgecolors[0],
                                      label=sample_labels[0], lw=2)
    s2 = ax[split_num - 2][1].scatter([], [], s=200, marker=shapes[1], facecolor=colors[1], edgecolor=edgecolors[1],
                                      label=sample_labels[1], lw=2)
    s3 = ax[split_num - 2][1].scatter([], [], s=100, marker='o', facecolor='None', edgecolor='black',
                                      label='Kroupa IMF',
                                      lw=2)
    s4 = ax[split_num - 2][1].scatter([], [], s=100, marker='d', facecolor='None', edgecolor='black',
                                      lw=2)

    handles = [s1, s2, (s3, s4, l1)]
    _, labels = ax[split_num - 2][1].get_legend_handles_labels()

    ax[split_num - 2][1].legend(handles=handles, labels=labels, fontsize=19,
                                handler_map={tuple: HandlerTuple(None)})

fig.subplots_adjust(wspace=0.23)
