import numpy as np


def bc03_rc15(band, color_name, color):
    color = np.array(color)
    if band not in ('g', 'r', 'i', 'z', 'H'):
        print("band must be one of 'grizH'")
    if color_name not in ('gr', 'gi', 'gz', 'ri', 'rz', 'rH', 'iz', 'iH', 'zH'):
        print("color_name must be one of 'gr,gi,gz,ri,rz,rH,iz,iH,zH'")
    if band == 'g':
        if color_name == 'gr':
            return 2.029 * color - 0.984
        if color_name == 'gz':
            return 1.116 * color - 1.132
        if color_name == 'rz':
            return 2.322 * color - 1.211
    if band == 'r':
        if color_name == 'gr':
            return 1.629 * color - 0.792
        if color_name == 'gz':
            return 0.9 * color - 0.916
        if color_name == 'rz':
            return 1.883 * color - 0.987
    if band == 'z':
        if color_name == 'gr':
            return 1.306 * color - 0.796
        if color_name == 'gz':
            return 0.716 * color - 0.888
        if color_name == 'rz':
            return 1.483 * color - 0.935


def b03_dm(band, grcolor):
    grcolor = np.array(grcolor)
    if band == 'r':
        return -0.306 + 1.097 * grcolor
    if band == 'z':
        return -0.272 + 0.699 * grcolor


def bc03_rc15_dm(band, grcolor):
    grcolor = np.array(grcolor)
    if band == 'r':
        return -0.792 + 1.629 * grcolor
    if band == 'z':
        return -0.781 + 1.308 * grcolor


def fsps_rc15_dm(band, grcolor):
    grcolor = np.array(grcolor)
    if band == 'r':
        return -0.647 + 1.497 * grcolor
    if band == 'z':
        return -0.619 + 1.120 * grcolor


def b01_jc_v(bvcolor):
    return -0.734 + 1.404 * bvcolor


def b19_etg_r(grcolor):
    return -0.59 + 1.35 * grcolor
