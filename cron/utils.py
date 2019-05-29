import netCDF4
import numpy as np
import logging
import os
from datetime import datetime, timedelta

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


# some needed fucntions
def wind_uv_to_dir(U, V):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the
    trig unit circle coordinate. If the wind directin is 360 then returns zero
    (by %360)
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WDIR = (270 - np.rad2deg(np.arctan2(V, U))) % 360
    return WDIR


def wind_uv_to_spd(U, V):
    """
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WSPD = np.sqrt(np.square(U) + np.square(V))
    return WSPD


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
        [[np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
          np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
         [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
          np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
         [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]])

    vel_matrix = np.asarray([[u], [v], [w]]).transpose()

    result = np.dot(vel_matrix, rot_matrix)

    return result


def calc_homogeneity(raw_vr, derived_vr):
    """
    Determines homogeneity of the wind field as described in E. Paschke et. al. 2015 section 2.2.4
    :param raw_vr: Raw radial velocity
    :param derived_vr: Radial velocity derived from wind retrieval
    :return:
    """

    vr_bar = np.sum(raw_vr)

    return 1 - np.sum((raw_vr - derived_vr) ** 2) / np.sum((raw_vr - vr_bar) ** 2)


def calc_vad_3d(az, elev, vel):
    """
    Calculates the 3D VAD
    :param az: Azimuth data
    :param elev: Elevation Data
    :param vel: Velocity Data
    :return:
        u: U component of the wind
        v: V component of the wind
        w: W component of the wind
    """
    elev = np.deg2rad(elev)
    az = np.deg2rad(az)

    if vel.size > 1:  # If there could be sufficient data points...
        A = sum(vel * np.sin(az))
        B = sum(np.sin(az) ** 2 * np.cos(elev))
        C = sum(np.cos(az) * np.sin(az) * np.cos(elev))
        G = sum(np.sin(az) * np.sin(elev))

        D = sum(vel * np.cos(az))
        E = sum(np.sin(az) * np.cos(az) * np.cos(elev))
        F = sum(np.cos(az) ** 2 * np.cos(elev))
        H = sum(np.cos(az) * np.sin(elev))

        W = sum(vel)
        X = sum(np.sin(az) * np.cos(elev))
        Y = sum(np.cos(az) * np.cos(elev))
        Z = sum(az * np.sin(elev))

        # solve A = uB + vC + wG , D = uE + vF + wH and W = uX + vY+ wZ
        y = np.array([[B, E, X], [C, F, Y], [G, H, Z]])
        z = np.array([A, D, W])
        # print y
        # print z
        try:
            sol = np.linalg.solve(y, z)
            # print sol
            u = sol[0]
            v = sol[1]
            w = sol[2]
            return u, v, w
        except np.linalg.linalg.LinAlgError:
            return FILL_VALUE, FILL_VALUE, FILL_VALUE
    else:
        return FILL_VALUE, FILL_VALUE, FILL_VALUE


def decode_header(header):
    """
    Takes in a list of lines from the raw hpl file. Separates them by
    tab and removes unnecessary text
    """
    new_header = {}

    for item in header:
        split = item.split('\t')
        new_header[split[0].replace(':', '')] = split[1].replace("\r\n", "")

    return new_header


def _to_epoch(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()


"""
process_file(in_file, out_dir, prefix):
Processes a raw halo hpl file and turns it into a netcdf
:param in_file:
:param out_dir:
:return:
"""


def writeVAD_to_nc(filename, date, elev, u, v, w, ws, wd, hgt, rmse, r_sq,up_flag,intensity):
    if os.path.exists(filename):
        # open the netcdf
        nc = netCDF4.Dataset(filename, 'a', format="NETCDF4")
        dim = nc.dimensions['t'].size

        u_var = nc.variables['u']
        v_var = nc.variables['v']
        w_var = nc.variables['w']
        ws_var = nc.variables['ws']
        wd_var = nc.variables['wd']
        rms_var = nc.variables['rms']
        r_sq_var = nc.variables['r_sq']
        time_var = nc.variables['time']
        hgt_var = nc.variables['hgt']
        up_flag_var = nc.variables['up_flag']

        u_var[dim, :] = u
        v_var[dim, :] = v
        w_var[dim, :] = w
        ws_var[dim, :] = ws
        wd_var[dim, :] = wd
        rms_var[dim, :] = rmse
        r_sq_var[dim, :] = r_sq
        up_flag_var[dim] = up_flag

    else:
        # Create the netcdf
        nc = netCDF4.Dataset(filename, 'w', format="NETCDF4")

        # Create the height dimension
        nc.createDimension('height', len(hgt))
        nc.createDimension('t', None)

        # Add the attributes
        nc.setncattr("elev", elev)
        nc.setncattr("date", date.isoformat())

        # Create the variables
        u_var = nc.createVariable('u', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        v_var = nc.createVariable('v', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        w_var = nc.createVariable('w', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        ws_var = nc.createVariable('ws', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        wd_var = nc.createVariable('wd', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        hgt_var = nc.createVariable('hgt', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        rms_var = nc.createVariable('rms', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        r_sq_var = nc.createVariable('r_sq', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        up_flag_var = nc.createVariable('up_flag', 'f8', ('t'))
        intensity_var = nc.createVariable('intensity', 'f8', ('t','hgt'))

        time_var = nc.createVariable('time', 'i8', ('t'))
        time_var.setncattr('units', 'seconds since 1970-01-01 00:00:00 UTC')
        dim = nc.dimensions['t'].size

    u_var[dim, :] = np.where(np.isnan(u), FILL_VALUE, u)
    v_var[dim, :] = np.where(np.isnan(v), FILL_VALUE, v)
    w_var[dim, :] = np.where(np.isnan(w), FILL_VALUE, w)
    ws_var[dim, :] = np.where(np.isnan(ws), FILL_VALUE, ws)
    wd_var[dim, :] = np.where(np.isnan(wd), FILL_VALUE, wd)
    hgt_var[dim, :] = np.where(np.isnan(hgt), FILL_VALUE, hgt)
    rms_var[dim, :] = np.where(np.isnan(rmse), FILL_VALUE, rmse)
    r_sq_var[dim, :] = np.where(np.isnan(r_sq), FILL_VALUE, r_sq)
    time_var[dim] = (date - datetime(1970, 1, 1)).total_seconds()
    up_flag_var[dim]=up_flag
    intensity_var[dim] = intensity

    # Close the netcdf
    nc.close()


def writeSTARE_to_nc(filename, date, w, hgt, intensity):

    logging.debug(filename)
    logging.debug(date)

    # Create the netcdf
    nc = netCDF4.Dataset(filename, 'w', format="NETCDF4")

    # Create the height dimension
    nc.createDimension('hgt', len(hgt))
    nc.createDimension('t', None)

    # Add the attributes
    nc.setncattr("date", date[0].isoformat())

    # Create the variables
    w_var = nc.createVariable('w', 'f8', ('t', 'hgt'), fill_value=FILL_VALUE)
    hgt_var = nc.createVariable('hgt', 'f8', ('hgt'), fill_value=FILL_VALUE)
    intensity_var = nc.createVariable('intensity', 'f8', ('t', 'hgt'))

    time_var = nc.createVariable('time', 'i8', ('t'))
    time_var.setncattr('units', 'seconds since 1970-01-01 00:00:00 UTC')

    hgt_var[:] = hgt
    time_var[:] = np.array([(d - datetime(1970, 1, 1)).total_seconds() for d in date])
    w_var[:, :] = w
    intensity_var[:, :] = intensity

    # Close the netcdf
    nc.close()


def writeRHI_to_nc(filename, date, vel, rng, elev, az, intensity,up_flag):
    if os.path.exists(filename):
        # open the netcdf
        nc = netCDF4.Dataset(filename, 'r+', format="NETCDF4")
        dim = nc.dimensions['t'].size

        vel_var = nc.variables['velocity']
        rng_var = nc.variables['range']
        elev_var = nc.variables['elevation']
        az_var = nc.variables['azimuth']
        intensity_var = nc.variables['intensity']
        time_var = nc.variables['time']
        up_flag_var = nc.variables['up_flag']

        # vel_var[dim, :] = vel
        # rng_var[:] = rng
        # elev_var[dim] = elev
        # az_var[dim] = az
        # intensity_var[dim, :] = intensity

    else:
        # Create the netcdf
        nc = netCDF4.Dataset(filename, 'w', format="NETCDF4")

        # Create the height dimension
        nc.createDimension('height', len(rng))
        nc.createDimension('t', None)

        # Create the variables
        vel_var = nc.createVariable('velocity', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        rng_var = nc.createVariable('range', 'f8', ('height'), fill_value=FILL_VALUE)
        elev_var = nc.createVariable('elevation', 'f8', ('t'), fill_value=FILL_VALUE)
        az_var = nc.createVariable('azimuth', 'f8', ('t'), fill_value=FILL_VALUE)
        intensity_var = nc.createVariable('intensity', 'f8', ('t', 'height'), fill_value=FILL_VALUE)
        up_flag_var = nc.createVariable('up_flag', 'f8', ('t'))

        time_var = nc.createVariable('time', 'i8', ('t'))
        time_var.setncattr('units', 'seconds since 1970-01-01 00:00:00 UTC')
        dim = nc.dimensions['t'].size

    dim2 = dim+len(date)
    vel_var[dim:dim2, :] = vel
    rng_var[:] = np.where(np.isnan(rng), FILL_VALUE, rng)
    elev_var[dim:dim2] = np.where(np.isnan(elev), FILL_VALUE, elev)
    az_var[dim:dim2] = np.where(np.isnan(az), FILL_VALUE, az)
    intensity_var[dim:dim2, :] = np.where(np.isnan(intensity), FILL_VALUE, intensity)
    time_var[dim:dim2] = [(d - datetime(1970, 1, 1)).total_seconds() for d in date]
    #print [up_flag for i in range(len(date))]
    up_flag_var[dim:dim2] = [float(up_flag) for i in range(len(date))]

    # Close the netcdf
    nc.close()
