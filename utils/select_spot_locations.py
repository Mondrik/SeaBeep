"""
select_spot_locations.py

Author: Nick Mondrik

--------------------
Given an input fits file location, display the file, and allow user to select spot locations.

To select a location, move the mouse over the spot and press "m".  The spot locations will be marked with a red circle.

After finishing, close the window, and the program will attempt to write the file.  If the input file is:
    /data/is/here/myfile.fits
then the data will be stored in:
    /data/is/here/here_spots.txt

This is compatible with the structure expected by run_phot.py

If the file mentioned above already exists, you will be asked if you wish to overwrite it.
If you specified -r or --replace, the file will automatically be (over-)written.
You may also use -o myfile or --outfile myfile to specify a specific output file location and name.

The locations saved in the file are transposed relative to the matplotlib coordinates because they are intended to be used to select elements in numpy array.  I.e., a point at (x=10, y=15) in a (20, 20) numpy array is accessed by my_array[15, 10], since numpy uses C indexing by default.
--------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import os
from cbp_helpers import findCenter as fc
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Use to select the location of CBP spots in the focal plane')
    parser.add_argument('filename', 
                        help='The name of the file from which to select the spot locations', type=str)
    parser.add_argument('-r', '--replace', action='store_true', help='If used, overwrite the spot file in location.  Otherwise, the program will ask before replacing the file')
    parser.add_argument('-o', '--outfile', help='If specified, write the output to this file instead of standard destination')

    return parser


spot_locs = []

def locate_spot(event):
    """
    Callback function for key_press_event from matplotlib.
    Stores the location of a spot the graph, for later use.
    Press "m" to store.
    """
    if event.key == 'm':
        canv = event.canvas
        # Have to manually reset focus to the figure or else it breaks after first use
        plt.figure(canv.figure.number)         
        ax = event.inaxes
        x = np.rint(event.xdata).astype(np.int)
        y = np.rint(event.ydata).astype(np.int)

        # append in opposite order because y = row, x = column
        adj_locs = fc(plt.gci().get_array().data, [[y, x]])
        spot_locs.append(adj_locs[0])
        ax.plot(spot_locs[-1][1], spot_locs[-1][0], 'ro')
        canv.draw()
        print('Point: ({:d}, {:d})'.format(spot_locs[-1][1], spot_locs[-1][0]))





def main(fitsfile, replace, outfile):
    """
    Open a matplotlib figure and listen for key_press_events.  Once all spots have been selected (with the "m" key), close the figure and the program will attempt to save the list of spot locations, either to the specified output file, or to X_spots.txt, where X is the root directory of the fits file.
    """
    global spot_locs
    d = pft.open(fitsfile)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(np.log10(d[0].data), origin='lower', vmin=3, vmax=4)
    plt.colorbar()
    cid = fig.canvas.mpl_connect('key_press_event', locate_spot)
    plt.show(block=True)
    spot_locs = np.array(spot_locs).astype(np.int)

    if outfile is None:
        # if outfile is none, build X_spots.txt filename from root directory name
        outfile_root = os.path.split(fitsfile)[0]
        run_name = os.path.split(outfile_root)[-1]
        outfile = os.path.join(outfile_root, run_name+'_spots.txt')

    if replace:
        np.savetxt(outfile, spot_locs, header='ROW COL', fmt='%d')
    else:
        if os.path.exists(outfile):
            print('File {} exists...  Overwrite? [y/[n]]'.format(outfile))
            while True:
                choice = input('>>>')
                if choice.lower() == 'y':
                    np.savetxt(outfile, spot_locs, header='ROW COL', fmt='%d')
                    print('{} overwritten'.format(outfile))
                    break
                elif choice.lower() == 'n' or choice.lower() == '':
                    print('{} not overwritten.'.format(outfile))
                    break
                else:
                    print('Input "{}" not recognized.  Please select y or n'.format(choice))
        else:
            np.savetxt(outfile, spot_locs, header='ROW COL', fmt='%d')
            print('{} has been created.'.format(outfile))

    return 

    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.filename, args.replace, args.outfile)
