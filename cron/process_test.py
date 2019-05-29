import json
import netCDF4
import numpy as np
import logging
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob

from utils import *

# from pyclamps.utils import list_to_masked_array, ray_height
# from pyclamps.vad import calc_vad_3d, calc_homogeneity

# Global Values
FILL_VALUE = -9999.
VEL_LIM = (-30, 30)
HGT_LIM = (0, 1000)
PROFILES_PER_PLOT = 2
Re = 6371000
R43 = Re * 4.0 / 3.0

# python logging NOT deployment logging
# logging.basicConfig(filename='DLprocessing.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


#########PROCESS CODE####################

#######deployment logging
# reads config file in TORUS_DL/logs/config.js
# writesout changes/events to deployment log in TORUS_DL/logs/log_MMDDYY.txt

# grab current timestamp for run
now = datetime.utcnow()
# now=datetime(2019,05,27,21,01)
log_time = now.strftime("%m%d%y_%H%M")
today = now.strftime("%Y%m%d")  # - commented out for test date below.
# today_l = datetime(2019,05,07)
# today=today_l.strftime("%Y%m%d")
# now = datetime(year=2019, month=5, day=7, hour=16, minute=15)

# open and read in config file info
config = open('/Users/elizabethsmith/TORUS_DL/logs/config.js')
logdata = json.load(config)
config.close()

if logdata["status"]=='up':
    print "we're up :)"
    up_flag=1
if logdata["status"]=='down':
    print "we're down :( "
    up_flag=0

# TB - Running into an error here when starting a new log file.
#      Added some automation to make your like easier

if os.path.exists('/Users/elizabethsmith/TORUS_DL/logs/' + logdata["logfile"]):
    # open and read most recent logfile entry
    current_logfile = open('/Users/elizabethsmith/TORUS_DL/logs/' + logdata["logfile"], "r+")
    lines = current_logfile.readlines()
    prev_status = lines[-6][8:-1]  # reading logfile previous status (skipping text)
    prev_heading = lines[-5][9:-1]  # reading logfile previous heading (skipping text)
    prev_lat = lines[-4][5:-1]  # reading logfile previous lat (skipping text)
    prev_lon = lines[-3][5:-1]  # reading logfile previous lat (skipping text)
    prev_note = lines[-2][6:-1]  # reading logfile previous note (skipping text)

    # check if the previous log entry matches the data in the config file..
    print logdata["note"], prev_note
    if (str(logdata["status"]) != prev_status or str(logdata["heading"]) != prev_heading or
                str(logdata["lat"]) != prev_lat or str(logdata["lon"]) != prev_lon or
                str(logdata["note"]) != prev_note):
        print '**CONFIG FILE HAS BEEN UPDATED!** generating log entry...'
        # generate writeout for new log entry based on config file.
        writeout = ["*********ENTRY**************\n",
                    "timestamp: %s\n" % log_time,
                    "status: %s\n" % logdata["status"],
                    "heading: %s\n" % logdata["heading"],
                    "lat: %s\n" % logdata["lat"],
                    "lon: %s\n" % logdata["lon"],
                    "note: %s\n" % logdata["note"],
                    "*********END***************\n"]
        current_logfile.writelines(writeout)
        print "**Logfile updated -- see /Users/elizabethsmith/TORUS_DL/logs/%s" % logdata["logfile"]
    else:
        print "--no config changes"
    current_logfile.close()
else:
    current_logfile = open('/Users/elizabethsmith/TORUS_DL/logs/' + logdata["logfile"], "w")
    writeout = ["*********ENTRY**************\n",
                "timestamp: %s\n" % log_time,
                "status: %s\n" % logdata["status"],
                "heading: %s\n" % logdata["heading"],
                "lat: %s\n" % logdata["lat"],
                "lon: %s\n" % logdata["lon"],
                "note: %s\n" % logdata["note"],
                "*********END***************\n"]
    current_logfile.writelines(writeout)
    print "**Logfile updated -- see /Users/elizabethsmith/TORUS_DL/logs/%s" % logdata["logfile"]


# get list of exisiting processed scans
path_proc = '/Users/elizabethsmith/TORUS_DL/data/nonQA_proc/dl/2019/201905/' + today + '/'

# check to make sure the output dir exists
try:
    os.makedirs(path_proc)
except OSError:
    logging.debug("Output path already exists...")

# Check to make sure the processed_files.txt exists
if not os.path.exists(path_proc + 'processed_files.txt'):
    os.system('touch {}'.format(path_proc + 'processed_files.txt'))

# Open the processed files list and read it in
proc_list = open(path_proc + 'processed_files.txt', "r+")
proc_files = proc_list.readlines()
proc_list.close()
# Be sure to add the files that are processed to the running list

# get list of existing raw scans - always do the stare
# TB - I changed some things here so only the scans from the current hour are even looked at.
#    - This cuts down on processing for the stare files!
path_raw = now.strftime('/Users/elizabethsmith/TORUS_DL/data/raw/dl/%Y/%Y%m/%Y%m%d/*%Y%m%d_*.hpl')
#print path_raw
raw_files = [f for f in glob(path_raw)]
#print raw_files
raw_files=sorted(raw_files)
# Process the scans
for in_file in raw_files:
    

    # TB - I changed your logic for finding the files to process. This is a little easier and less prone to bugs
    # Check to see if the file is in the alreasy processed files. If it is, skip it.
    if in_file+'\n' in proc_files:
        logging.debug("{} already processed".format(in_file))
        continue
    else:
        logging.info("Processing {}".format(in_file))
        

    # read in new scan
    out_dir = path_proc
    prefix = 'nonQA'

    # Read in the text file
    lines = []
    with open(in_file) as f:
        for line in f:
            lines.append(line)

    logging.debug("Decoding header")
    # Read in the header info
    header = decode_header(lines[0:11])

    ngates = int(header['Number of gates'])
    # nrays = int(header['No. of rays in file'])  # Cant do this apparently. Not always correct (wtf)
    len_data = len(lines[17:])
    nrays = len_data / (ngates + 1)

    gate_length = float(header['Range gate length (m)'])
    start_time = datetime.strptime(header['Start time'], '%Y%m%d %H:%M:%S.%f')
    scan_type = None

    logging.debug("Reading data")
    # Read in the actual data
    az = np.zeros(nrays)
    hour = np.zeros(nrays)
    elev = np.zeros(nrays)
    pitch = np.zeros(nrays)
    roll = np.zeros(nrays)
    rng = np.asarray([(gate + .5) * gate_length for gate in range(ngates)])

    vel = np.zeros((ngates, nrays))
    intensity = np.zeros((ngates, nrays))
    beta = np.zeros((ngates, nrays))

    try:
        for ray in range(nrays):
            # Get the scan info
            info = lines[ray * (ngates + 1) + 17].split()
            hour[ray] = float(info[0])
            az[ray] = float(info[1])
            elev[ray] = float(info[2])
            pitch[ray] = float(info[3])
            roll[ray] = float(info[4])

            for gate in range(ngates):
                data = lines[ray * (ngates + 1) + 17 + gate + 1].split()
                vel[gate, ray] = float(data[1])
                intensity[gate, ray] = float(data[2])
                beta[gate, ray] = float(data[3])

    except IndexError:
        logging.warning("Something went wrong with the indexing here...")

    # correction for some rounding/hysteresis in scanner azimuths... setting all vals==360. to 0.
    az[np.where(az == 360.)] = 0.

    # dynamic identification of lidar scan type (fp,ppi,rhi)
    # TB - I Added the round here. Was getting a fp file ID'd as a rhi file
    #    - Also had an issue with and RHI file where az[0] was 0.01 and az[2] was 0
    try:
        if np.round(az[0], 1) == np.round(az[2], 1):  # const azimuth could be RHI or stare
            if np.round(elev[0], 1) == np.round(elev[2], 1):  # const azimuth and constant elev = stare
                scan_type = 'fp'

            else:  # const azimuth and non-constant elev = RHI
                scan_type = 'rhi'

        elif np.round(elev[0], 1) == np.round(elev[2]):  # changing azimuth, const elev = PPI
            scan_type = 'ppi'

        if scan_type == None:
            raise IndexError

        logging.info("Scan Type: " + scan_type)

    except IndexError:
        logging.warning("Something went wrong with scan type IDing...")

    if scan_type == 'ppi':
        date = datetime.strptime(start_time.strftime('%Y-%m-%dT%H:%M:%S'), "%Y-%m-%dT%H:%M:%S")
        hgt = []
        u = []
        v = []
        w = []
        rmse = []
        r_sq = []
        intensity_out = []
        for i, rng in enumerate(rng):
            # Get the required stuff for this range ring
            cnr = intensity[i, :]  # range,azimuth
            Vel = vel[i, :]  # range,azimuth
            I = intensity[i, :] # range,azimuth
            Az = az  # 8-terms of az
            Elev = elev  # 8-terms of az

            # Filter out the bad values based on CNR - default was 1.015
            Az = np.where(cnr <= 1.01, FILL_VALUE, Az) 
            Vel = np.where(cnr <= 1.01, FILL_VALUE, Vel)

            Az = list_to_masked_array(Az, FILL_VALUE)
            Vel = list_to_masked_array(Vel, FILL_VALUE)

            # Calculate the vad and height for this range ring
            tmp_u, tmp_v, tmp_w = calc_vad_3d(Az, Elev, Vel)  # grab this to it can point to it!!!

            # Calculate the RMSE
            N = float(Vel.size)
            az_rad = np.deg2rad(Az)
            elev_rad = np.deg2rad(Elev)

            derived_vr = (np.sin(az_rad) * np.cos(elev_rad) * tmp_u) + \
                         (np.cos(az_rad) * np.cos(elev_rad) * tmp_v) + \
                         (np.sin(elev_rad) * tmp_w)

            tmp_E = Vel - derived_vr

            # Calculate rms error
            tmp_RMSE = np.sqrt(1 / N * np.sum(tmp_E ** 2))

            tmp_r_sq = calc_homogeneity(Vel, derived_vr)

            # Append to the lists for plotting
            u.append(tmp_u)
            v.append(tmp_v)
            w.append(tmp_w)
            hgt.append(ray_height(rng, Elev[0]))
            rmse.append(tmp_RMSE)
            r_sq.append(tmp_r_sq)
            intensity_out.append(np.mean(I))
        vector_wind = rotate(u, v, w, logdata["heading"], 0, 0)
        vector_wind = vector_wind.squeeze()
        u = vector_wind[:, 0]
        v = vector_wind[:, 1]
        w = vector_wind[:, 2]
        ws = wind_uv_to_spd(u, v)
        wd = wind_uv_to_dir(u, v)

        writeVAD_to_nc(path_proc + prefix + date.strftime('%Y%m%d') + '_VAD.nc', date, elev, u, v, w, ws, wd, hgt,
                       rmse, r_sq,up_flag, intensity_out)

        # add newly processed file to list of processed files
        proc_list = open(path_proc + 'processed_files.txt', "a")
        proc_list.writelines(in_file+'\n')
        proc_list.close()

    if scan_type == 'fp':

        # TB - I decided that it is best to just process the entire stare file every time
        # instead of try to append to the netcdf. This shouldn't hinder processeing time that much
        # since I changed things to only grab things from the same hour you're processing

        date = datetime.strptime(start_time.strftime('%Y-%m-%dT%H:%M:%S'), "%Y-%m-%dT%H:%M:%S")
        times = np.asarray([datetime(year=date.year, month=date.month, day=date.day) + timedelta(hours=h) for h in hour])

        # Filter out the bad values based on CNR
        Vel = np.where(intensity <= 1.01, FILL_VALUE, vel)

        logging.debug("Writing stare file")
        writeSTARE_to_nc(path_proc+prefix+date.strftime('%Y%m%d_%H_STARE.nc'), times,
                         vel.transpose(), rng, intensity.transpose())

    if scan_type=='rhi':
        # TB - A quick tip: Don't do an RHI at az=0. It bounces between 0 and 360 and is a pain in the ass to process
        #    - Just do it at like 1 deg. or even .1

        # TB - Note to self - need to do a heading correction on this one.

        date = start_time
        times = np.asarray([datetime(year=date.year, month=date.month, day=date.day) + timedelta(hours=h) for h in hour])
        filename = path_proc + prefix + date.strftime('%Y%m%d_%H') + '_RHI.nc'
        # break
        writeRHI_to_nc(filename, times, vel.transpose(), rng, elev, az, intensity.transpose(), up_flag)

        # add newly processed file to list of processed files
        proc_list = open(path_proc + 'processed_files.txt', "a")
        proc_list.writelines(in_file+'\n')
        proc_list.close()
