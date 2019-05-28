"""
Collection of utility functions for processing CLAMPS data
"""
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray
import smtplib
import os
from datetime import datetime

from netCDF4 import Dataset
from numpy import sin, cos
from pint import UnitRegistry
from scipy.optimize import leastsq
from scipy.signal import correlate


from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import COMMASPACE, formatdate
from email import Encoders
import ConfigParser

from pyclamps.tower import Tower


# Constants
Re = 6371000
R43 = Re * 4.0 / 3.0

PREV_CROSS_SECTIONS = {}


def geo2sph(x, y, z):
    '''
    Convert cartesian (geographic) coridinates to spherical coordinates
    :param x: X Coordinate
    :param y: Y Coordinate
    :param z: Z Coordinate
    :return:
        r: Range
        elev: Elevation
        az: Azimuth
    '''
    r = np.sqrt(x**2 + y**2 + z**2)               # r
    elev = np.arctan2(z, np.sqrt(x**2 + y**2))     # theta
    az = np.arctan2(-x, -y)                           # phi
    return r, elev, az


def get_terrain_cross_section(terrain_nc, lat_0, lon_0, az, rng):
    '''

    :param terrain_nc:
    :param lat_0:
    :param lon_0:
    :param az:
    :param rng:
    :return:
    '''
    # If we already have the data for this cross section...
    if az in PREV_CROSS_SECTIONS.keys():
        return PREV_CROSS_SECTIONS[az]

    # Otherwise do the process....
    # Open the netcdf
    nc = Dataset(terrain_nc)

    # Get the projection
    proj = pyproj.Proj(nc['x'].proj4)

    # Convert the lats/lons to UTM coords
    x, y = proj(nc['lon'][:], nc['lat'][:])

    # Get the location of interest
    x_0, y_0 = proj(lon_0, lat_0)

    # Convert the coordinates relative to x_0, y_0
    x_rel = x - x_0
    y_rel = y - y_0

    # Calculate points of the cross section
    x_cross = rng * np.sin(np.deg2rad(az))
    y_cross = rng * np.cos(np.deg2rad(az))
    z_cross = []

    for i in range(rng.size):
        ind = np.argmin(np.abs(np.sqrt((x_rel - x_cross[i]) ** 2. + (y_rel - y_cross[i]) ** 2.)))
        z_cross.append(nc['z'][:].flatten()[ind])

    z_cross = np.asarray(z_cross)
    z = nc['z'][:]
    nc.close()

    # Add the data to the previous datasets
    PREV_CROSS_SECTIONS[az] = (z_cross, (x_cross, y_cross), (x_rel, y_rel, z))

    return z_cross, (x_cross, y_cross), (x_rel, y_rel, z)


def get_tower_from_ucar_nc(nc_file, t_id):
    """
    Extracts tower data from the netcdf based on tower ID.
    :param nc_file: Filename of the tower netcdf
    :param t_id: ID of the tower (ex 'tnw01')
    :return: Tower object containing the data
    """

    # Open the netcdf
    nc = Dataset(nc_file)

    # Create a unit registry to make sure everything is in base units and to correctly read in data.
    ureg = UnitRegistry()

    # Init a dict to hold the tower information
    tower = Tower(t_id, nc['latitude_'+t_id][0], nc['longitude_'+t_id][0])

    # Don't forget the tower time
    times = [datetime.fromtimestamp(d) for d in nc['base_time'][:] + nc['time'][:]]
    tower.time = np.asarray(times)

    # Loop through all the keys to find the ones with the correct tower id
    for key in nc.variables.keys():
        if t_id in key:
            try:

                # Split the key to get all the info needed
                info = key.split('_')

                # Only grab info from keys formatted like var_height_tower
                # TODO - Account for radiation and flux data
                if len(info) == 3:
                    tower.add_measurement(info[0], ureg(info[1]).to_base_units().magnitude, nc[key][:])
                elif len(info) == 5:
                    tower.add_measurement('{}_{}'.format(info[0], info[1]), ureg(info[3]).to_base_units().magnitude, nc[key][:])

            except AttributeError:
                pass

    nc.close()
    return tower


def jet_max(t, z, ws, buf=12):
    # need time, height  (assumes constant dz), and wind speed, buffer of hieght for max to move in time
    print('Finding max...')
    z_llj = np.full_like(t, 0)  # store jet max heights in time
    ws_llj = np.full_like(t, 0)  # store jet max ws in time
    for i in range(0, len(t)):
        wsmax = np.nanmax(ws[i, :22])
        if np.isnan(wsmax):
            z_llj[i] = np.nan
            ws_llj[i] = np.nan
        else:
            wsmax_index = np.where(ws[i, :22] == wsmax)[0][0]
            z_llj[i] = z[wsmax_index]
            ws_llj[i] = wsmax
            ##TIM METHOD######
            #   dz  = z[1]-z[0]
            #   for i in range(0,len(t)):
            #       shear = np.full_like(ws[i,:],np.nan)
            #       for j in range(0,len(z)-1):
            #           shear[j] = (ws[i,j+1]-ws[i,j])/dz
            #       if np.all(np.isnan(shear)):
            #           max_index = 0
            #       else:
            #           max_index = np.where(shear<(-.001))[0][0]-1 #after Tim's threshold, bonin 2015 in BLM
            #       #print max_index
            # z_llj[i] = z[max_index]
            #       ws_llj[i] = ws[i,max_index]

    return z_llj, ws_llj


def list_to_masked_array(in_list, mask_value):
    a = np.array(in_list)
    return np.ma.masked_where(a == mask_value, a)


def ray_height(rng, elev, H0=0, R1=R43):
    """
    Center of radar beam height calculation.
    Rinehart (1997), Eqn 3.12, Bech et al. (2003) Eqn 3
    INPUT::
    -----
    r : float
        Range from radar to point of interest [m]
    elev : float
        Elevation angle of radar beam [deg]
    H0 : float
        Height of radar antenna [m]
    R1 : float
        Effective radius
    OUTPUT::
    -----
    H : float
        Radar beam height [m]
    USAGE::
    -----
    H = ray_height(r,elev,H0,[R1=6374000.*4/3])
    NOTES::
    -----
    If no Effective radius is given a "standard atmosphere" is assumed,
       the 4/3 approximation.
    Bech et al. (2003) use a factor ke that is the ratio of earth's radius
       to the effective radius (see r_effective function) and Eqn 4 in B03
    """

    # Convert earth's radius to km for common dN/dH values and then
    # multiply by 1000 to return radius in meters
    hgt = np.sqrt(rng ** 2 + R1 ** 2 + 2 * rng * R1 * np.sin(np.deg2rad(elev)))
    hgt = hgt - R1 + H0

    return hgt


def rotate(u, v, w, yaw, pitch, roll):
    '''
    Calculate the value of u, v, and w after a specified axis rotation
    :param u: U component of the wind
    :param v: V component of the wind
    :param w: W component of the wind
    :param yaw: Rotation about the Z axis
    :param pitch: Rotation about the X axis
    :param roll: Rotation about the Y axis
    :return:
        result: 3D array of the new U, V, and W fields after the rotation
    '''

    rot_matrix = np.asarray(
        [[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
        [sin(yaw)*cos(pitch) , sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
        [-sin(pitch)         , cos(pitch)*sin(roll)                            , cos(pitch)*cos(roll)]])

    vel_matrix = np.asarray([[u], [v], [w]]).transpose()

    result = np.dot(vel_matrix, rot_matrix)

    return result


def send_mail(send_from, send_to, subject, text, files=None,
              data_attachments=None, server="smtp.mail.me.com", port=587,
              tls=True, html=False, images=None,
              username=None, password=None,
              config_file=None, config=None):
    '''
    Send an email to the specified email addresses
    :param send_from: "from" email address
    :param send_to: List of addresses to send the email to
    :param subject: Subject of the email
    :param text: Text of the email
    :param files: Path to files to attach to the email
    :param data_attachments:
    :param server: Server to send the email from
    :param port: Port on the server to send the email from
    :param tls:
    :param html:
    :param images:
    :param username:
    :param password:
    :param config_file:
    :param config:
    :return:
    '''

    if files is None:
        files = []

    if images is None:
        images = []

    if data_attachments is None:
        data_attachments = []

    if config_file is not None:
        config = ConfigParser.ConfigParser()
        config.read(config_file)

    if config is not None:
        server = config.get('smtp', 'server')
        port = config.get('smtp', 'port')
        tls = config.get('smtp', 'tls').lower() in ('true', 'yes', 'y')
        username = config.get('smtp', 'username')
        password = config.get('smtp', 'password')

    msg = MIMEMultipart('related')
    msg['From'] = send_from
    msg['To'] = send_to if isinstance(send_to, basestring) else COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach( MIMEText(text, 'html' if html else 'plain') )

    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(f,"rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
        msg.attach(part)

    for f in data_attachments:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( f['data'] )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % f['filename'])
        msg.attach(part)

    for (n, i) in enumerate(images):
        fp = open(i, 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        msgImage.add_header('Content-ID', '<image{0}>'.format(str(n+1)))
        msg.attach(msgImage)

    smtp = smtplib.SMTP(server, int(port))
    if tls:
        smtp.starttls()

    if username is not None:
        smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

