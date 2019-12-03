import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import os
import argparse


spot_locs = []

def locate_spot(event):
    if event.dblclick:
        x = np.rint(event.xdata).astype(np.int)
        y = np.rint(event.ydata).astype(np.int)
        ax = event.inaxes
        ax.plot(x, y, 'ro')
        plt.gcf().canvas.draw()
        print('Point: ({:d}, {:d})'.format(x, y))
        # append in opposite order because y = row, x = column
        spot_locs.append([y, x])


def create_parser():
    parser = argparse.ArgumentParser(description='Use to select the location of CBP spots in the focal plane')
    parser.add_argument('filename', nargs=1, 
                        help='The name of the file from which to select the spot locations', type=str)
    parser.add_argument('-r', '--replace', action='store_true', help='If used, overwrite the spot file in location.  Otherwise, the program will ask before replacing the file')

    return parser


def main(fitsfile, replace):
    global spot_locs
    d = pft.open(fitsfile)
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(np.log10(d[0].data), origin='lower', vmin=3, vmax=4)
    plt.colorbar()
    cid = fig.canvas.mpl_connect('button_press_event', locate_spot)
    plt.show(block=True)
    
    spot_locs = np.array(spot_locs).astype(np.int)
    outfile_root = os.path.split(fitsfile)[0]
    run_name = os.path.split(outfile_root)[-1]
    outfile_name = os.path.join(outfile_root, run_name+'_spots.txt')
    if replace:
        np.savetxt(outfile_name, spot_locs, header='ROW COL')
    else:
        if os.path.exists(outfile_name):
            print('File {} exists...  Overwrite? [y/[n]]'.format(outfile_name))
            while True:
                a = input('>>>')
                if a.lower() == 'y':
                    np.savetxt(outfile_name, spot_locs, header='ROW COL')
                    print('File overwritten')
                    break
                elif a.lower() == 'n' or a.lower() == '':
                    print('File not overwritten.')
                    break
                else:
                    print('Input {} not recognized.  Please select y or n')
        else:
            print('File {} has been created.'.format(outfile_name))

    return 

    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    fitsfile = args.filename[0]
    replace = args.replace
    main(fitsfile, replace)
