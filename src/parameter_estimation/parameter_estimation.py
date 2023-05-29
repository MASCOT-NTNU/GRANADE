import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp
from shapely.geometry import Polygon, Point
from scipy import interpolate
import imageio
import time
from datetime import datetime
from datetime import date
from math import degrees, radians
from numpy import vectorize
from pathlib import Path


from src.WGS import WGS


# Get the list of all files and directories
path = os.getcwd() + "/src/sinmod_files/"
dir_list = os.listdir(path)
sinmod_files = []
for file in dir_list:
    if file.split(".")[-1] == "nc":
        sinmod_files.append(file)

print(sinmod_files)


# This is the file number we want
file_num = 6
sinmod = netCDF4.Dataset(path + sinmod_files[file_num])
sinmod_file_name = sinmod_files[file_num][:-3]

timestamp = sinmod['time']
time_stamps = np.array(timestamp)
lat = np.array(sinmod['gridLats'])
lon = np.array(sinmod['gridLons'])
lats = lat.reshape(-1,1)
lons = lon.reshape(-1,1)

depth = np.array(sinmod['depth'])
elevation = np.array(sinmod['elevation'])

salinity = np.array(sinmod['salinity'])
temperature = np.array(sinmod["temperature"])