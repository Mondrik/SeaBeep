
import os, sys, glob
import optparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (13, 8) if False else (10, 6)

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--logDir",default="/data/nas/CTIO_0p9m/october2017/logs_20171002")
    parser.add_option("--doPlots",  action="store_true", default=False)

    opts, args = parser.parse_args()
 
    return opts
 
# Parse command line
opts = parse_commandline()
name = opts.logDir.split("/")[-1]
xmlfiles = glob.glob(os.path.join(opts.logDir,'*.xml'))

filename = os.path.join("output","combine_%s.dat"%name)
fid = open(filename,'w')
for xmlfile in xmlfiles:
    name = xmlfile.split("/")[-1].replace(".xml","")
    #if True:
    if not os.path.isfile("output/%s/photons.dat"%(name)):
        if opts.doPlots:
            system_command = "python run_log_p9m.py -f %s --doPlots"%(xmlfile)
            print system_command
            os.system(system_command)
        else:
            system_command = "python run_log_p9m.py -f %s"%(xmlfile)
            print system_command
            os.system(system_command)

    data_out = np.loadtxt("output/%s/photons.dat"%(name)) 
    fid.write('%s %.5e %.5e\n'%(name,data_out[0],data_out[1]))
fid.close()

