import numpy as np
import matplotlib.pyplot as plt

def get_cbp_transmission(wl_grid, native=False):
    wl, wpa = np.loadtxt('../data/cbp_cal_WperA.txt',unpack=True)

    h = 6.62607e-34 #  m^2 kg / s
    c = 2.99792e8 #  m / s
    echarge = 1.602177e-19 #  Coulombs

    wl_meters = wl*1e-9 #  wavelength in meters
    e_photon = h*c/wl_meters #  Joules/photon
    qe = wpa/e_photon*echarge

    if native:
        return wl, qe
    else:
        interp = np.interp(wl_grid, wl, qe)
        return interp
