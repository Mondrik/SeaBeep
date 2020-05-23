import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def get_cbp_transmission(wl_grid, native=False, uncert_region_size=10):

    # Load relevant test data from NIST
    telcbp = np.loadtxt('../data/tel-cbp.txt')
    #telmon = np.loadtxt('../data/tel-monitor.txt')
    #ts04mon = np.loadtxt('../data/ts04-monitor.txt')
    telts04 = np.loadtxt('../data/tel-ts04.txt')
    ts04aperw = np.loadtxt('../data/ts04-aperw.txt')

    # First, we need to smooth the tel/ts04 data by low pass filtering it:
    # the data is sampled every nm, so the nyquist frequency is 0.5/1nm = 0.5 nm^-1
    # The digital butterworth filter cutoff frequency is defined relative to the
    # nyquist freq., so a cutoff of 0.05 corresponds to 0.05*0.5 nm^-1 = 0.025 nm^-1
    # Also, since we use a double pass filter (to avoid phasing issues), the
    # order of the filter is effectively doubled (so order=3 corresponds to 6th order in total)
    telts04_smoothed = lowpass_filter(telts04[:,1])

    # The most limited wavelength set is ts04 A/W, so we have to interpolate everything to that.
    wl = ts04aperw[:,0]
    telts04_interp = np.interp(wl, telts04[:,0], telts04_smoothed)
    tcbp = np.interp(wl, telcbp[:,0], telcbp[:,1])

    # CBP W/A is given by:
    wpa = tcbp/telts04_interp/ts04aperw[:,1]

    # We use the non-smoothed version to estimate the uncertainty:
    raw_wpa_interp = np.interp(wl, telts04[:,0], telts04[:,1])
    raw_wpa = tcbp/raw_wpa_interp/ts04aperw[:,1]

    # Convert from W/A to Quantum Efficiency
    h = 6.62607e-34 #  m^2 kg / s
    c = 2.99792e8 #  m / s
    echarge = 1.602177e-19 #  Coulombs

    wl_meters = wl*1e-9 #  wavelength in meters
    e_photon = h*c/wl_meters #  Joules/photon
    transmission = wpa/e_photon*echarge
    raw_transmission = raw_wpa/e_photon*echarge

    # Estimate uncertainty as stddev in 10 nm bin:
    transmission_uncert = estimate_cbp_uncert(wl, transmission, raw_transmission, local_region=uncert_region_size)

    if native:
        return wl, transmission, transmission_uncert
    else:
        interp = np.interp(wl_grid, wl, transmission, left=np.nan, right=np.nan)
        uncert_interp = np.interp(wl_grid, wl, transmission_uncert, left=np.nan, right=np.nan)
        return interp, uncert_interp


def lowpass_filter(data, order=3, cutoff=0.05):
    b, a = signal.butter(order, cutoff)
    return signal.filtfilt(b, a, data)


def estimate_cbp_uncert(wl_grid, smoothed, raw, local_region=10):
    """
    Since we lack a formal error budget for the CBP transmission measurements (due to the interference fringes in the refracting telescope, we estimate an uncertainty as the standad deviation in a local region about each measurement point.
    """
    
    uncerts = np.zeros_like(wl_grid)

    resid = raw - smoothed
    interp_grid = np.arange(np.min(wl_grid), np.max(wl_grid)+1, 1)
    interp_resid = np.interp(interp_grid, wl_grid, resid)

    for i,w in enumerate(wl_grid):
        interp_idx = int(w - np.min(interp_grid))
        start = interp_idx - int(local_region/2)
        stop = interp_idx + int(local_region/2) + 1
        if start < 0:
            start = 0
            stop += (local_region - stop)
        elif stop > len(interp_grid):
            start -= (stop - len(interp_grid))
            stop = len(interp_grid)

        #print(f'Calculating uncerts from {interp_grid[start]} to {interp_grid[stop-1]} for lambda={w}')
        uncerts[i] = np.std(interp_resid[start:stop])
        
    # Per Joe and Ping, uncertainty on the CBP/tel, tel/monitor, monitor/ts04, R_trap quantities are
    # 0.1%, 0.5%, 0.5%, 0.1%, for 400 nm <= lambda < 700 nm
    # 0.1%, 1%, 1%, 0.1% for lambda >= 700 nm.
    region_1 = np.where(np.logical_and(wl_grid >= 400, wl_grid < 700))[0]
    uncert_1 = smoothed[region_1]**2.*(0.001**2. + 0.005**2. + 0.005**2. + 0.001**2.) 

    region_2 = np.where(wl_grid >= 700)[0]
    uncert_2 = smoothed[region_2]**2.*(0.001**2. + 0.01**2. + 0.01**2. + 0.001**2.)

    uncerts[region_1] = np.sqrt(uncerts[region_1]**2. + uncert_1)
    uncerts[region_2] = np.sqrt(uncerts[region_2]**2. + uncert_2)

    return uncerts
