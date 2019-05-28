import logging
import numpy as np
import os
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import cmocean
from netCDF4 import Dataset

# Get the current time
now = datetime.utcnow()

# Configure the log
# logging.basicConfig(filename='DLprocessing.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Some plotting paramters for later
prefix = 'RHI_nonQA'
VMIN = -20
VMAX = 20
HGT_LIM = (0, 1200)
XLIM = (-1200, 1200)
YLIM = (0, 1000)

# Get the RHI file for this hour
fname = glob(now.strftime('../data/nonQA_proc/dl/%Y/%Y%m/%Y%m%d/*%Y%m%d_%H_RHI.nc'))[0]

# Open the netcdf
logging.debug("Opening netcdf")
nc = Dataset(fname, 'r')

scans = []
counter = 0

'''use this section if there are multiple azimuths'''
# Figure out how many scans there are
# last = None
# for az in nc['azimuth'][:]:
#     if last is None:
#         last = az
#     elif az != last:
#         last = az
#         scans.append(counter)
#         counter += 1
#     else:
#         scans.append(counter)

'''use this section if there is only one azimuth. Assumes elevations are increasing'''
# Figure out how many scans there are
last = None
for elev in nc['elevation'][:]:
    if last is None:
        pass
    elif elev < last:
        scans.append(counter)
        counter += 1
    else:
        scans.append(counter)
    last = elev

scans = np.asarray(scans)

for scan in range(0, counter+1):
    ind = np.where(scans == scan)

    # Get the date
    time = datetime.utcfromtimestamp(nc['time'][ind][0])
    logging.debug("Processing RHI: " + time.isoformat())
    
    outdir = time.strftime('../plots/%Y%m%d/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = time.strftime('../plots/%Y%m%d/{prefix}_%Y%m%d_%H%M%S.png'.format(prefix=prefix))

    if os.path.exists(outfile):
        continue
    else:
        logging.info("Processing " + outfile)

    # Get the grid figured out
    elev = np.deg2rad(nc['elevation'][ind])
    rng_m = nc['range'][:]
    az = np.round(np.mean(np.where(nc['azimuth'][ind] == 360., 0, nc['azimuth'][ind])))

    # Get the data
    vel = nc['velocity'][ind]
    intensity = nc['intensity'][ind]
    # vel = np.ma.masked_where(intensity < 1.01, vel)

    elev, rng_m = np.meshgrid(elev, rng_m)

    x_m = rng_m * np.cos(elev)
    y_m = rng_m * np.sin(elev)

    # if az >= 180.: x_m *= -1.

    #     vel = vel[sort, :].transpose()
    vel = vel.transpose()

    # Get the axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Make the plot
    c = ax.pcolor(x_m, y_m, vel, vmin=VMIN, vmax=VMAX, cmap=cmocean.cm.delta)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.colorbar(c)
    plt.title('RHI {} {}'.format(az, time.isoformat()))

    plt.savefig(outfile)
    plt.savefig('../plots/RHI_latest.png')

nc.close()


