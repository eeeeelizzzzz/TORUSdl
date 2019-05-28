# Standard Libraries
import csv
from datetime import datetime, timedelta

# 3rd Party Libs
import numpy as np
import netCDF4
import xarray


def concat_files(files, concat_dim='time'):
    """
    concatenates files based on a certain dimension and returns the data in a dictionary
    :param files: Files to concatenate
    :param concat_dim: Dimension to concatenate alone
    :return: Dictionary of  var_name:data_array
    """
    arr = xarray.open_mfdataset(files, autoclose=True, concat_dim=concat_dim, decode_times=False)
    data = {}

    for key in arr.keys():
        data[key] = arr[key].values
        data[key] = np.ma.masked_where(np.isnan(data[key]), data[key])

    arr.close()

    return data


def get_aeri_variables(fn, variables):
    """
    Extract specific variables and mask areas of questionable quality. time and height will
    be extracted automatically. This will also mask out large gaps in data
    :param fn: Filename
    :param variables: List of variables to extract
    :return: Dict of data
    """
    # Open the dataset
    nc = netCDF4.Dataset(fn)

    # Make the dictionary to store the variables. Start with time and height
    data = {'time': np.asarray([datetime.utcfromtimestamp(d) for d in (nc['base_time'][:]+nc['time_offset'][:])]),
            'height': nc['height'][:]}

    # Make qc flag into the same shape as the data
    qc_flag = np.meshgrid(nc['qc_flag'], data['height'])[0].transpose()

    # Need to find gaps larger than 10 minutes so they can be blanked out correctly
    times = np.asarray([datetime.utcfromtimestamp(d) for d in (nc['base_time'][:]+nc['time_offset'][:])])
    new_times = []
    gaps_counter = 0
    for i in range(times.size)[1:]:
        # If there's larger than a 10 minute gap in the data, add it in so we can mask it out later
        if times[i] - times[i-1] > timedelta(seconds=10*60):
            num_missing = (times[i] - times[i-1]).total_seconds() / (10*60)
            for num in range(1, int(num_missing)):
                new_times.append(times[i]+timedelta(seconds=num*10*60))
                gaps_counter += 1
    times = np.append(times, new_times)

    # create bad data array to append to the good data.
    bad = np.zeros(shape=(gaps_counter, data['height'].size))
    bad[:] = np.nan

    # Get the sorting order based on time and add the sorted times to the data dictionary
    sort_order = np.argsort(times)
    data['time'] = times[sort_order]

    for var in variables:
        # Make sure the variable is there
        if var not in nc.variables.keys():
            raise KeyError("{} not in aeri variables list".format(var))

        tmp = nc[var][:]
        # Mask values with bad qc
        tmp = np.ma.MaskedArray(np.append(tmp, bad, axis=0))
        tmp = tmp[sort_order]
        # mask the gaps
        tmp.mask = np.isnan(tmp)
        data[var] = tmp

    # Add in the qc variable
    bad = np.zeros(shape=(gaps_counter, data['height'].size))
    bad[:] = 1
    qc_flag = np.ma.MaskedArray(np.append(qc_flag, bad, axis=0))
    qc_flag = qc_flag[sort_order]
    data['qc_flag'] = qc_flag

    nc.close()

    return data


def read_wind_profile(profile):
    """
    Reads in a wind profile processed by the Halo lidar
    :param profile: File to process
    :return:
    """

    with open(profile, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)

        for i, line in enumerate(reader):
            if i == 0:
                num_hgt = int(line[0])
                z = np.zeros(num_hgt)
                dir = np.zeros(num_hgt)
                spd = np.zeros(num_hgt)
            else:
                z[i-1] = line[0]
                dir[i-1] = line[1]
                spd[i-1] = line[2]

    return z, dir, spd
