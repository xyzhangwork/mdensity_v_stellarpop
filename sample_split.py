from astropy.io import fits
import numpy as np
from astropy.table import Table
from sklearn.ensemble import IsolationForest
from scipy import odr
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def target_function(p, x):
    m, c = p
    return m * x + c


odr_model = odr.Model(target_function)

if __name__ == '__main__':
    parameters = fits.open('parameters_reff2_2.fits')
    gals = parameters[1].data['plateifu']
    split = {'plateifu': gals,
             'm20kpc_v_sigmacen': [parameters[1].data['Mout_20kpc'],
                                   parameters[1].data['sigma_cen'],
                                   np.full(len(gals), np.nan)],
             'mtot_v_m10kpc': [parameters[1].data['mass_sga_rc15'],
                               parameters[1].data['Min_10kpc'],
                               np.full(len(gals), np.nan)],
             'mtot_v_m20kpc': [parameters[1].data['mass_sga_rc15'],
                               parameters[1].data['Mout_20kpc'],
                               np.full(len(gals), np.nan)],
             'sigmacen_v_m20kpc': [parameters[1].data['sigma_cen'],
                                   parameters[1].data['Mout_20kpc'],
                                   np.full(len(gals), np.nan)],
             'm10kpc_v_m20kpc': [parameters[1].data['Min_10kpc'],
                                 parameters[1].data['Mout_20kpc'],
                                 np.full(len(gals), np.nan)],
             'm10kpc_v_m20kpc_control': [parameters[1].data['Min_10kpc'],
                                         parameters[1].data['Mout_20kpc'],
                                         np.full(len(gals), np.nan)],
             'mre_v_m2re': [parameters[1].data['Min_re'],
                            parameters[1].data['Mout_2re'],
                            np.full(len(gals), np.nan)]
             }
    sample_bin = [[1, 35], [65, 99]]
    extendedness = np.full(len(gals), np.nan)
    highsig = np.full(len(gals), np.nan)

    for type_name in [*split][1:]:
        dx = split[type_name][0]
        dy = split[type_name][1]
        mask_nan0 = (~np.isnan(dx)) & (~np.isnan(dy))
        var = np.full(len(dx), np.nan)

        X0 = np.array([dx[mask_nan0], dy[mask_nan0]]).T
        clf = IsolationForest(random_state=0).fit(X0)
        mask = clf.decision_function(X0) > np.nanpercentile(clf.decision_function(X0), 3)
        mask_nan = mask & mask_nan0
        X = np.array([dx[mask_nan], dy[mask_nan]]).T

        if type_name == 'm20kpc_v_sigmacen':
            z = np.polyfit(X[:, 0], X[:, 1], 1)
            p = np.poly1d(z)
            var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]
            highsig[mask_nan0] = (dy[mask_nan0] - p(dx[mask_nan0])) / np.sqrt(p.coeffs[0] ** 2 + 1)
        elif type_name == 'm10kpc_v_m20kpc':# or (type_name == 'mre_v_m2re'):
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(scale(X))
            X0_reduced = pca.fit_transform(scale(X0))

            pca = PCA(n_components=2).fit(X_reduced)
            var[mask_nan] = X_reduced[:, 0] * (pca.components_[0][1] / pca.components_[0][0]) - X_reduced[:, 1]
            if type_name == 'm10kpc_v_m20kpc':
                extendedness[mask_nan0] = X0_reduced[:, 0] * (pca.components_[0][1] / pca.components_[0][0]) - X0_reduced[:,
                                                                                                           1]
        else:
            z = np.polyfit(X[:, 0], X[:, 1], 1)
            p = np.poly1d(z)
            var[mask_nan] = p(dx[mask_nan]) - dy[mask_nan]

        for i in range(2):
            mask_tmp = (var <= np.nanpercentile(var[mask_nan], sample_bin[i][1])) & \
                       (var >= np.nanpercentile(var[mask_nan], sample_bin[i][0]))
            split[type_name][2][mask_tmp] = int(np.nanmean(dy[mask_tmp]) <= np.nanmean(dy))  # + type_name_i * 2
            print(np.nanpercentile(var[mask_nan], sample_bin[i][1]), np.nanpercentile(var[mask_nan], sample_bin[i][0]))

    dx0 = parameters[1].data['sigma_cen'][split['m10kpc_v_m20kpc'][2] == 0]
    dx1 = parameters[1].data['sigma_cen'][split['m10kpc_v_m20kpc'][2] == 1]
    bins_p = np.histogram_bin_edges(parameters[1].data['sigma_cen'], bins='auto')
    cnt0, _ = np.histogram(dx0, bins=bins_p)
    cnt1, _ = np.histogram(dx1, bins=bins_p)

    cat0 = np.array([], dtype='<U11')
    cat1 = np.array([], dtype='<U11')
    for i in range(len(cnt0)):
        cat_tmp2 = gals[
            (parameters[1].data['sigma_cen'] >= bins_p[i]) & (parameters[1].data['sigma_cen'] <= bins_p[i + 1])]
        if cnt0[i] > cnt1[i]:
            cat_tmp = np.intersect1d(gals[split['m10kpc_v_m20kpc'][2] == 0], cat_tmp2)
            cat0 = np.insert(cat0, 0, np.random.choice(cat_tmp, size=cnt1[i], replace=False))
            cat1 = np.insert(cat1, 0, np.intersect1d(gals[split['m10kpc_v_m20kpc'][2] == 1], cat_tmp2))
        else:
            cat_tmp = np.intersect1d(gals[split['m10kpc_v_m20kpc'][2] == 1], cat_tmp2)
            cat0 = np.insert(cat0, 0, np.intersect1d(gals[split['m10kpc_v_m20kpc'][2] == 0], cat_tmp2))
            cat1 = np.insert(cat1, 0, np.random.choice(cat_tmp, size=cnt0[i], replace=False))
    _, idxx_0, _ = np.intersect1d(gals, cat0, return_indices=True)
    _, idxx_1, _ = np.intersect1d(gals, cat1, return_indices=True)
    split['m10kpc_v_m20kpc_control'][2][idxx_0] = 0
    split['m10kpc_v_m20kpc_control'][2][idxx_1] = 1
    names = list(split.keys())
    names.append('extendedness')
    names.append('highsig')

    t = Table([split['plateifu'], split['m20kpc_v_sigmacen'][2], split['mtot_v_m10kpc'][2],
               split['mtot_v_m20kpc'][2], split['sigmacen_v_m20kpc'][2], split['m10kpc_v_m20kpc'][2],
               split['m10kpc_v_m20kpc_control'][2], split['mre_v_m2re'][2],extendedness, highsig],
              names=names)
    t.write('sample_split_outliers_total2.fits', overwrite=True)
