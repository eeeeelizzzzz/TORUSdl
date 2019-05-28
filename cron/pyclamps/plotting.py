# System Libraries
import os

# MPL gets special treatment for serverside processing
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# 3rd party Libraries
import cmocean
import numpy as np
import scipy.interpolate as interpolate
from colormaps import create_colormap
from colormap2d import get_cmap2d

# From this library
from .utils import jet_max, get_terrain_cross_section
from pyclamps.__init__ import perdigao_clamps_lat, perdigao_clamps_lon

# Make some colormaps for later
cm_ws = create_colormap(1000, base='ncl_helix', name='helix', reverse=False, white=False)
cm_bias = create_colormap(1000, base='ncl_temp_diff_18lev', name='tempdiff', reverse=False, white=False)
cm_wd = ListedColormap(get_cmap2d('darkwheel')[:, -1])
z_0 = 0

cmaps = {
    'w':  {'cm': 'seismic',   'label': 'vertical velocity [m/s]'},
    'ws': {'cm': 'gist_stern_r',              'label': 'windspeed [m/s]'},
    # 'ws': {'cm': 'CMRmap_r',              'label': 'windspeed [m/s]'},
    'wd': {'cm': cm_wd,   'label': 'wind direction [deg]'},
    # 'wd': {'cm': cmocean.cm.phase,   'label': 'wind direction [deg]'},
    'pt': {'cm': cmocean.cm.thermal, 'label': 'potential temperature [C]'},
    'q':  {'cm': cmocean.cm.haline_r,  'label': 'q [g/kg]'},
    'dp': {'cm': cmocean.cm.haline_r,  'label': 'dewpoint [C]'},
    'rh': {'cm': cmocean.cm.haline_r,  'label': 'RH [%]'},
    'std': {'cm': cmocean.cm.thermal,  'label': 'Standard Deviation'}
}


def rhi_plot(elev, rng_m, vel, az, time, vmin=-5, vmax=5,
             xlim=(-7500, 7500), ylim=(0, 7500), terrain_file=None,
             lat_0=perdigao_clamps_lat, lon_0=perdigao_clamps_lon):
    '''
    Function that plots clamps rhi data.
    :param elev: 1-D elevation data
    :param rng_m: 1-D range range data
    :param vel: 2-d velocity data (dims: elevation, range)
    :param az: Azimuth of data
    :param time: datetime object of rhi time
    :param vmin: Minimum velocity
    :param vmax: Maximum velocity
    :param xlim: X limit of plot
    :param ylim: Y limit of plot
    :param terrain_file: Netcdf file containing elevation data
    :param lat_0: Latitiude of CLAMPS
    :param lon_0: Longitude of CLAMPS
    :return:
        ax: matplotlib axis object containing the plot
        z_0: Elevation of CLAMPS extracted from the terrain file (if not None)
    '''

    # TODO - Take out all the terrain plotting stuff and make that happen in the script

    elev, rng_m = np.meshgrid(elev, rng_m)

    x_m = rng_m * np.cos(elev)
    y_m = rng_m * np.sin(elev)

    # if az >= 180.: x_m *= -1.

    #     vel = vel[sort, :].transpose()
    vel = vel.transpose()

    # Get the axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # See if we need to do things with the terrain
    z_0 = None
    if terrain_file is not None:
        cross_ranges = np.arange(-2000, 2000, 100)
        terr_elev, cross_pts, elev_grid = get_terrain_cross_section(terrain_file, lat_0,
                                                                    lon_0, az, cross_ranges)
        cross_section = interpolate.interp1d(cross_ranges, terr_elev)
        z_0 = cross_section(0.)
        y_m += z_0
        ax.plot(cross_ranges, terr_elev)

    # Make the plot
    c = ax.pcolor(x_m, y_m, vel, vmin=vmin, vmax=vmax, cmap=cmocean.cm.delta)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(c)
    plt.title('RHI {} {}'.format(az, time.isoformat()))

    # Create the inset if the terrain file was used
    if terrain_file is not None:
        x1, y1, x2, y2 = [cross_pts[0][0], cross_pts[1][0], cross_pts[0][-1], cross_pts[1][-1]]
        inset = inset_axes(ax, width=1., height=1., loc=2)
        inset.pcolormesh(elev_grid[0], elev_grid[1], elev_grid[2], vmin=200, vmax=500)
        inset.set_xlim(-2000, 2000)
        inset.set_ylim(-2000, 2000)
        inset.arrow(x1, y1, x2 - x1, y2 - y1, color='r')
        #         inset.plot(cross_pts[0], cross_pts[1], color='r')
        inset.get_xaxis().set_visible(False)
        inset.get_yaxis().set_visible(False)

    return ax, z_0


def time_height(time, height, data, field, ax=None, datemin=None, datemax=None,
                datamin=None, datamax=None, zmin=None, zmax=None):
    '''
    Produces a time height plot of a 2-D field
    :param time: Array of times (1-D or 2-D but must have same dimenstions as height)
    :param height: Array of heights (1-D or 2-D but must have same dimensions as time)
    :param data: Array of the data to plot (2-D)
    :param field: Field being plotted. Currently supported:
        'w': Vertical Velocity
        'ws': Wind Speed
        'wd': Wind Direction
        'pt': Potential Temperature
        'q':  Specific Humidity
        'dp': Dewpoint
        'rh': Relative Humidity
        'std': Standard Deviation

    :param ax: Axis to plot the data to
    :param datemin: Datetime object
    :param datemax: Datetime object
    :param datamin: Minimum value of data to plot
    :param datamax: Maximum value of data to plot
    :param zmin: Minimum height to plot
    :param zmax: Maximum height to plot
    :return:
    '''

    # Get the colormat and label of the data
    cm, cb_label = cmaps[field]['cm'], cmaps[field]['label']

    # If there isn't an axis already, create one to use
    if ax is None:
        fig, ax = plt.subplots(1)

    # Convert the dates to matplolib format if not done already
    if time.ndim == 1 and height.ndim == 1:
        time = mdates.date2num(time)
        time, height = np.meshgrid(time, height)

    # Create the plot
    c = ax.pcolormesh(time, height, data, vmin=datamin, vmax=datamax, cmap=cm)

    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax)
    cb.set_label(cb_label)

    # Format the limits
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if zmin is not None and zmax is not None:
        ax.set_ylim(zmin, zmax)
    if datemin is not None and datemax is not None:
        ax.set_xlim(mdates.date2num(np.array([datemin, datemax])))

    # Set the labels
    ax.set_ylabel('Height [m]')
    # ax.set_xlabel('Time [UTC]')

    return ax


def vad_component_plot(u, v, w, hgt, rmse, r_sq, date, prefix, out_dir,
                       vel_lim=(-30, 30), hgt_lim=(0, 1000)):
    '''
    Plots U, V, W, rmse, and R^2 profiles with height
    :param u:
    :param v:
    :param w:
    :param hgt:
    :param rmse:
    :param r_sq:
    :param date:
    :param prefix:
    :param out_dir:
    :param vel_lim:
    :param hgt_lim:
    :return:
    '''
    # TODO - Make this use an axis object instead of creating it's own
    fig = plt.figure(1, figsize=(25, 7))
    fig.suptitle(date.isoformat())
    plt.subplot(1, 5, 1)
    plt.plot(u, hgt, label=date.strftime("%H:%M"))
    plt.title("U velocity vs Height")
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Height (m)')
    plt.xlim(vel_lim)
    plt.ylim(hgt_lim)
    plt.legend()

    plt.subplot(1, 5, 2)
    plt.plot(v, hgt)
    plt.title("V velocity vs Height")
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Height (m)')
    plt.xlim(vel_lim)
    plt.ylim(hgt_lim)

    plt.subplot(1, 5, 3)
    plt.plot(w, hgt)
    plt.title("W velocity vs Height")
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Height (m)')
    plt.xlim((-5, 5))
    plt.ylim(hgt_lim)

    plt.subplot(1, 5, 4)
    plt.plot(rmse, hgt)
    plt.title("RMS vs Height")
    plt.xlabel('RMS')
    plt.ylabel('Height (m)')
    plt.xlim((0, 10))
    plt.ylim(hgt_lim)

    plt.subplot(1, 5, 5)
    plt.plot(r_sq, hgt)
    plt.title("$R^2$ vs Height")
    plt.xlabel('$R^2$')
    plt.ylabel('Height (m)')
    plt.xlim((.8, 1))
    plt.ylim(hgt_lim)

    # Create profile base on start time of first profile
    image_name = "{prefix}_{date}.png"
    image_name = image_name.format(prefix=prefix, date=date.strftime("%Y%m%d_%H%M%S"))
    image_name = os.path.join(out_dir, image_name)

    # Save the image
    plt.savefig(image_name)
    plt.clf()
    plt.close()