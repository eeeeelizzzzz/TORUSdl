import logging
import os
from datetime import datetime, timedelta
from glob import glob
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cmocean as cm
import numpy as np
# Get the current time
now = datetime.utcnow()
#now=datetime(2019,05,17,22,58)

# current_day=now.day
# if now.hour>1:
#     first_hour = str(current_day)+'_'+str(now.hour-2)
#     second_hour = str(current_day)+'_'+str(now.hour-1)
#     jump_day=False
# else:
#     first_hour = str(current_day-1)+'_'+str(24+(now.hour-2))
#     second_hour = str(current_day-1)+'_'+str(24+(now.hour-1))
#     jump_day=True

tmp = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
XLIM = (mdates.date2num(tmp-timedelta(hours=1)), mdates.date2num(tmp+timedelta(hours=1)))

# Configure the log
# logging.basicConfig(filename='DLprocessing.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Some plotting paramters for later
prefix = 'VAD_nonQA'
prefix1 = 'HODO_nonQA'
prefix2 = 'VADtz_nonQA'
VEL_LIM = (0, 40)
HGT_LIM = (0, 2000)
# Get the VAD file for this hour
fnames = glob(now.strftime('../data/nonQA_proc/dl/%Y/%Y%m/%Y%m%d/*%Y%m%d*VAD.nc'))
for fname in sorted(fnames):
    # Open the netcdf
    logging.debug("Opening netcdf")
    nc = Dataset(fname)
    
    # Extract the data
    logging.debug("Getting data")
    u = nc['u'][:]
    v = nc['v'][:]
    ws = nc['ws'][:]
    wd = nc['wd'][:]
    r_sq = nc['r_sq'][:]
    hgt = nc['hgt'][:]
    times = [datetime.utcfromtimestamp(d) for d in nc['time'][:]]
    


    for i, time in enumerate(times):
        out_dir = time.strftime('../plots/%Y%m%d/')
        outfile = time.strftime(out_dir + '{prefix}_%Y%m%d_%H%M%S.png'.format(prefix=prefix))
        if os.path.exists(outfile):
            logging.debug(outfile + ' already exists! Skipping...')
            continue
        else:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logging.info("Plotting VAD file: " + outfile)
    
        fig, (ws_ax, wd_ax, r_sq_ax) = plt.subplots(1, 3)
        fig.set_figheight(10)
        fig.set_figwidth(15)
    
        ws_ax.plot(ws[i], hgt[i], 'k-*',linewidth=2,markersize=10)
        if i>2:
            ws_ax.plot(ws[i-1], hgt[i-1], 'k-*',alpha=0.5)
            ws_ax.plot(ws[i-2], hgt[i-2], 'k-*',alpha=0.2)
        ws_ax.set_ylim(HGT_LIM)
        ws_ax.set_xlim(VEL_LIM)
        ws_ax.set_ylabel("Height AGL (m)")
        ws_ax.set_xlabel("Wind Speed (m/s)")
        ws_ax.grid()
    
        wd_ax.plot(wd[i], hgt[i], 'k*',markersize=10)
        if i>2:
            wd_ax.plot(wd[i-1], hgt[i-1], 'k*',alpha=0.5)
            wd_ax.plot(wd[i-2], hgt[i-2], 'k*',alpha=0.2)
        wd_ax.set_ylim(HGT_LIM)
        wd_ax.set_xlim(0, 360)
        wd_ax.set_ylabel("Height AGL (m)")
        wd_ax.set_xlabel("Wind Direction (degrees)")
        wd_ax.grid()
    
        r_sq_ax.plot(r_sq[i], hgt[i],'k',linewidth=2)
        if i>2:
            r_sq_ax.plot(r_sq[i-1], hgt[i-1],'k',alpha=0.5)
            r_sq_ax.plot(r_sq[i-2], hgt[i-2],'k',alpha=0.2)
        r_sq_ax.set_ylim(HGT_LIM)
        r_sq_ax.set_xlim(.5, 1)
        r_sq_ax.set_ylabel("Height AGL (m)")
        r_sq_ax.set_xlabel("R Squared")
        r_sq_ax.grid()
    
        plt.suptitle(time.isoformat())
    
        plt.savefig(outfile)
        plt.savefig("../plots/VAD_latest.png")
        plt.close()
            
    #hodos
    for i, time in enumerate(times):
        out_dir = time.strftime('../plots/%Y%m%d/')
        outfile = time.strftime(out_dir + '{prefix}_%Y%m%d_%H%M%S.png'.format(prefix=prefix1))
        if os.path.exists(outfile):
            logging.debug(outfile + ' already exists! Skipping...')
            continue
        else:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logging.info("Plotting hodograph from VAD: " + outfile)
    
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        
        ax.axis([-20,20,-20,20])
        for direction in ["left","bottom"]:
            ax.spines[direction].set_position('zero')
            ax.spines[direction].set_smart_bounds(True)
        for direction in ["right","top"]:
            ax.spines[direction].set_color('none')
        ax.set_xticks(np.concatenate([np.arange(-20,0,5),np.arange(5,21,5)]))
        ax.set_yticks(np.concatenate([np.arange(-20,0,5),np.arange(5,21,5)]))
        t=np.linspace(0,2*np.pi,360)
        for rad in range(5,21,5):
            plt.plot(rad*np.cos(t),rad*np.sin(t),'k--',lw=0.5)
        #sys.exit()
        for c,lev in zip(['red','orange','green','blue','magenta','black'],[0,200,400,600,800,1000]):
            uu=u[i,np.where(hgt[0,:]>lev)[0]]
            vv=v[i,np.where(hgt[0,:]>lev)[0]]
            plt.plot(uu,vv,color=c,lw=1.5)
        
        plt.suptitle(time.isoformat())
    
        plt.savefig(outfile)
        plt.savefig("../plots/VADhodo_latest.png")
        plt.close()
    
    outfile = tmp.strftime(out_dir + '{prefix}_%Y%m%d_%H%M%S.png'.format(prefix=prefix2))
    
    times, hgt = np.meshgrid(times, hgt[0])
    
    # Make the plot
    logging.info("Plotting t-zplot from VAD: " + outfile)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(10)
    fig.set_figwidth(15)
        
    c = ax1.pcolormesh(mdates.date2num(times), hgt, ws.transpose(), vmin=0, vmax=40, cmap='magma_r')
    
    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax1)
    cb.set_label('horizontal wind speed [m/s]')
    
    # Format the limits
    ax1.set_ylabel('Height [m]')
    ax1.set_xlabel('Time [UTC]')
    ax1.set_xlim(XLIM)
    ax1.set_ylim(HGT_LIM)
    
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    ax1.set_title('Generated: {}'.format(now.isoformat()))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    c = ax2.pcolormesh(mdates.date2num(times), hgt, wd.transpose(), vmin=0, vmax=360, cmap=cm.cm.phase)
    
    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax2)
    cb.set_label('wind direction [deg]')
    
    # Format the limits
    ax2.set_ylabel('Height [m]')
    ax2.set_xlabel('Time [UTC]')
    ax2.set_xlim(XLIM)
    ax2.set_ylim(HGT_LIM)
    
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    ax2.set_title('Generated: {}'.format(now.isoformat()))
    
    
    plt.tight_layout()
    plt.savefig(outfile)
    plt.savefig('../plots/VADtz_latest.png')
