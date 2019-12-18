"""
find_altaz_spot_derivs.py

Author: Nick Mondrik

--------------------
Given a pair of input fits files with differing CBP alt/az or mount pointings, calculate (delta_pix / delta_angle) where angle is CBP alt/az or mount alt/az

To select a location, move the mouse over the spot and press "m".  The spot locations will be marked with a red circle.  The location of the spot is found by convolving the image with a large (20 pix) gaussian, and taking the maximum of the resulting smmothed image in a region around the cursor.

After finishing, close the window, and the program will attempt to write the file.  If the input file is:
    /data/is/here/myfile.fits
then the data will be stored in:
    /data/is/here/here_spot_derivs.pkl

This is compatible with the structure expected by run_phot.py

If the file mentioned above already exists, you will be asked if you wish to overwrite it.
If you specified -r or --replace, the file will automatically be (over-)written.
You may also use -o myfile or --outfile myfile to specify a specific output file location and name.
--------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import os
from cbp_helpers import findCenter as fc
import argparse
import pickle


def create_parser():
    parser = argparse.ArgumentParser(description='Used to find the derivative of the spot location on the focal plane w.r.t. CBP or telescope mount axes')
    parser.add_argument('filenames', 
                        help='The name of the files from which to select the spot locations.  Files should have different CBP alt/az or mount pointings', type=str, nargs=2)
    parser.add_argument('-r', '--replace', action='store_true', help='If used, overwrite the spot file in location.  Otherwise, the program will ask before replacing the file')
    parser.add_argument('-o', '--outfile', help='If specified, write the output to this file instead of standard destination')
    parser.add_argument('-a', '--axis', help='Which motor axis derivative to measure. E.g., cbpmountalt, cbpmountaz, mountalpha, mountdelta')

    return parser


spot_locs_one = []
spot_locs_two = []

def locate_spot(event):
    """
    Callback function for key_press_event from matplotlib.
    Stores the location of a spot the graph, for later use.
    Press "m" to store.
    """

    global fignum_one
    global fignum_two
    if event.key == 'm':
        canv = event.canvas
        # Have to manually reset focus to the figure or else it breaks after first use
        plt.figure(canv.figure.number)         
        ax = event.inaxes
        x = np.rint(event.xdata).astype(np.int)
        y = np.rint(event.ydata).astype(np.int)

        # append in opposite order because y = row, x = column
        adj_locs = fc(plt.gci().get_array().data, [[y, x]])
        if canv.figure.number == fignum_one:
            spot_locs_one.append(adj_locs[0])
        elif canv.figure.number == fignum_two:
            spot_locs_two.append(adj_locs[0])
        else:
            print('Key press was detected in figure number {}, but only figure numbers {} and {} are valid.  Close the program and try again?'.format(canv.figure.number, fignum_one, fignum_two))
            raise ValueError

        ax.plot(adj_locs[-1][1], adj_locs[-1][0], 'ro')
        canv.draw()
        print('Point: ({:d}, {:d}) Figure: {}'.format(adj_locs[-1][1], adj_locs[-1][0], canv.figure.number))





def main(fitsfiles, replace, outfile, axis):
    """
    Open two matplotlib figures and listen for key_press_events.  Once all spots have been selected in both figures (with the "m" key), close the figures and the program will attempt to save the list of spot locations, either to the specified output file, or to X_spots_deriv.txt, where X is the root directory of the fits file.
    """
    
    global spot_locs_one
    global spot_locs_two
    global fignum_one
    global fignum_two
    
    d1 = pft.open(fitsfiles[0])
    d2 = pft.open(fitsfiles[1])

    # Set up first figure:
    fig_one = plt.figure(figsize=(12, 12))
    plt.title(os.path.split(fitsfiles[0])[-1])
    plt.xlabel('X')
    plt.ylabel('Y')
    fignum_one = fig_one.number
    plt.imshow(np.log10(d1[0].data), origin='lower', vmin=3, vmax=4)
    plt.colorbar()
    cid = fig_one.canvas.mpl_connect('key_press_event', locate_spot)

    # Set up second figure:
    fig_two = plt.figure(figsize=(12, 12))
    plt.title(os.path.split(fitsfiles[1])[-1])
    plt.xlabel('X')
    plt.ylabel('Y')
    fignum_two = fig_two.number
    plt.imshow(np.log10(d2[0].data), origin='lower', vmin=3, vmax=4)
    plt.colorbar()
    cid = fig_two.canvas.mpl_connect('key_press_event', locate_spot)

    plt.show(block=True)

    spot_locs_one = np.array(spot_locs_one).astype(np.int)
    spot_locs_two = np.array(spot_locs_two).astype(np.int)
    delta_angle = float(d1[0].header[axis]) - float(d2[0].header[axis])

    deriv = (spot_locs_one - spot_locs_two) / delta_angle
    print(deriv)
    print(np.mean(deriv, axis=0))

    if outfile is None:
        # if outfile is none, build X_spots.txt filename from root directory name
        outfile_root = os.path.split(fitsfiles[0])[0]
        run_name = os.path.split(outfile_root)[-1]
        outfile = os.path.join(outfile_root, run_name+'_spot_derivs.pkl')

    if replace:
        if os.path.exists(outfile):
            mydict = pickle.load(open(outfile, 'rb'))
        else:
            mydict = {}
        mydict[axis] = np.mean(deriv, axis=0)
        with open(outfile, 'wb') as myfile:
            pickle.dump(mydict, myfile)
    else:
        if os.path.exists(outfile):
            print('File {} exists...  Overwrite? [y/[n]]'.format(outfile))
            while True:
                choice = input('>>>')
                if choice.lower() == 'y':
                    mydict = pickle.load(open(outfile, 'rb'))
                    mydict[axis] = np.mean(deriv, axis=0)
                    with open(outfile, 'wb') as myfile:
                        pickle.dump(mydict, myfile)
                    print('{} overwritten'.format(outfile))
                    break
                elif choice.lower() == 'n' or choice.lower() == '':
                    print('{} not overwritten.'.format(outfile))
                    break
                else:
                    print('Input "{}" not recognized.  Please select y or n'.format(choice))
        else:
            mydict = {}
            mydict[axis] = np.mean(deriv, axis=0)
            with open(outfile, 'wb') as myfile:
                pickle.dump(mydict, myfile)
            print('{} has been created.'.format(outfile))

    return 

    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.filenames, args.replace, args.outfile, args.axis)
