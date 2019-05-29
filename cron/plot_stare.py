import logging
import os
from datetime import datetime, timedelta
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from netCDF4 import Dataset

# Get the current time
now = datetime.utcnow()
#now=datetime(2019,05,17,23,58)

# Configure the log
# logging.basicConfig(filename='DLprocessing.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Some plotting paramters for later
prefix = 'Stare_nonQA'
VMIN = -4
VMAX = 4
HGT_LIM = (0, 2000)

tmp = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
XLIM = (mdates.date2num(tmp-timedelta(hours=0)), mdates.date2num(tmp+timedelta(hours=1)))


# current_day=now.day
# if now.hour>1:
#     first_hour = str(current_day)+'_'+str(now.hour-2)
#     second_hour = str(current_day)+'_'+str(now.hour-1)
#     jump_day=False
# else:
#     first_hour = str(current_day-1)+'_'+str(24+(now.hour-2))
#     second_hour = str(current_day-1)+'_'+str(24+(now.hour-1))
#     jump_day=True

# Get the STARE file for this hour
fnames = glob(now.strftime('../data/nonQA_proc/dl/%Y/%Y%m/%Y%m%d/*%Y%m%d_*_STARE.nc'))
for fname in sorted(fnames):
    outdir = now.strftime('../plots/%Y%m%d/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outfile = now.strftime('../plots/%Y%m%d/{prefix}_%Y%m%d_%H.png'.format(prefix=prefix))

    # Open the most recent netcdf
    logging.debug("Opening netcdf")
    nc = Dataset(fname)
    
    # Get the data
    w = nc['w'][:]
    time = np.array([datetime.utcfromtimestamp(d) for d in nc['time'][:]])
    hgt = nc['hgt'][:]

    nc.close()
    
    # White out the spaces where there is not any stare data
    ind = np.where((time[1:] - time[:-1]) > timedelta(seconds=10))
    w[ind] = 0

    # Make the mesh grid
    time = mdates.date2num(time)
    time, hgt = np.meshgrid(time, hgt)
    
    # Make the plot
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    
    c = ax.pcolormesh(time, hgt, w.transpose(), vmin=-4, vmax=4, cmap='seismic')
    
    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax)
    cb.set_label('vertical velocity [m/s]')
    
    # Format the limits
    ax.set_ylabel('Height [m]')
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(XLIM)
    ax.set_ylim(HGT_LIM)
    
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    ax.set_title('Generated: {}'.format(now.isoformat()))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.savefig('../plots/stare_latest.png')
