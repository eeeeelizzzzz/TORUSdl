import numpy as np

# TODO - Find a different way to manage times
# TODO - Work in different types of measurements (radiation, wind, thermo, etc)
# TODO - Work in Flux Calculations with flux tower object
# TODO - Work in averaging and filtering
# TODO - Work in units


class Measurement(object):
    def __init__(self, name, height, data):
        self.name = name
        self.heights = [height]
        self.data = {height: np.asarray(data)}

    def __str__(self):
        return str(self.data)

    def __getitem__(self, height):
        return self.data[height]

    def add_level(self, height, new_data):

        # If the height is already here
        if height in self.heights:
            raise RuntimeError("Height '{}' already in Measurement {}".format(height, self.name))

        # Add the new data
        self.heights.append(height)
        self.data[height] = new_data


class Tower(object):

    def __init__(self, name, lat=None, lon=None, time=None):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.time = time

        # Dict for storage of measurements
        self.measurements = {}

    def __str__(self):
        tmp = "Tower ID: {name}\n " \
              " Latitude: {lat}, Longitude: {lon}\n" \
              " Measurements: {vars}"

        return tmp.format(name=self.name, lat=self.lat, lon=self.lon, vars=self.measurements.keys())

    def __getitem__(self, item):
        return self.measurements[item]

    def add_measurement(self, name, height, data):
        # See if this measurement is already registered for the tower
        if name not in self.measurements.keys():
            measurement = Measurement(name, height, data)
            self.measurements[name] = measurement

        else:
            self.measurements[name].add_level(height, data)

    def remove_measurement(self, name):
        del self.measurements[name]





