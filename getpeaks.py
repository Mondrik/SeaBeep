import astropy.io.fits as pft
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import glob


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = np.ones((35,35))
    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image<5*np.median(image))

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = ~ (local_max ^ eroded_background)

    return detected_peaks


def reject_isolated(peaks):
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            if peaks[i,j] is False:
                continue
            else:
                try:
                    above = peaks[i+1,j]
                    below = peaks[i-1,j]
                    left = peaks[i,j-1]
                    right = peaks[i,j+1]
                    if (not above) and (not below) and (not left) and (not right):
                        peaks[i,j] = False
                except IndexError as e:
                    peaks[i,j] = False
    return peaks

def find_centers(peaks):
    centers = []
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            if peaks[i,j] is False:
                continue
            else:
                try:
                    above = peaks[i+1,j]
                    below = peaks[i-1,j]
                    left = peaks[i,j-1]
                    right = peaks[i,j+1]
                    if (above) and (below) and (left) and (right) and not peaks[i,j]:
                        centers.append([i,j])
                except IndexError as e:
                    peaks[i,j] = False
    return np.array(centers)


def get_spot_guesses(image):
    peaks = detect_peaks(image)
    peaks = reject_isolated(peaks)
    centers = find_centers(peaks)
    return centers