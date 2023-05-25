"""
Prior module for the GRANADE project.

Objective:
    - Create prior data used for the Planner.
"""

from WGS import WGS

import os
import netCDF4
import numpy as np
import time
from pathlib import Path
from scipy import interpolate
from datetime import datetime



class Prior: 

    def __init__(self, sinmod_path_str) -> None:

        # Reading the sinmod file
        # folderpath = os.getcwd() + "/../sinmod/"
        # files = os.listdir(folderpath)
        # sinmod_path_str = ""
        # for file in files:
        #     if file.endswith(".nc"):
        #         sinmod_path_str = folderpath + files[0]
        # if sinmod_path_str == "":
        #     raise Exception("No sinmod file found")

        self.sinmod_path = sinmod_path_str
        self.sinmod_data = netCDF4.Dataset(sinmod_path_str)
       
        

        self.n_depts: int
        self.n_timesteps: int
        self.start_date: str
        self.start_second: float

        # Loading sinmod data to an 
        self.sinmod_data_dict = {}
        self.__load_sinmod_data()

        t1 = time.time()  #REMOVE
        self.interpolation_functions = self.__make_interpolation_functions()
        t2 = time.time()  #REMOVE
        print(f"== Create interpolate functions == t={t2-t1:,.2f} s")  #REMOVE


    def __load_sinmod_data(self) -> None:
        t1 = time.time()
        
        # Getting the times 
        self.sinmod_data_dict["timestamp"] = self.sinmod_data['time']
        self.sinmod_data_dict["time_stamps"] = np.array(self.sinmod_data_dict["timestamp"])

        # Coordinates
        self.sinmod_data_dict["lat"] = np.array(self.sinmod_data['gridLats'])
        self.sinmod_data_dict["lon"] = np.array(self.sinmod_data['gridLons'])
        self.sinmod_data_dict["lats"] = self.sinmod_data_dict["lat"].reshape(-1,1)
        self.sinmod_data_dict["lons"] = self.sinmod_data_dict["lon"].reshape(-1,1)

        # Converte to xy
        y,x = WGS.latlon2xy(self.sinmod_data_dict["lats"], self.sinmod_data_dict["lons"])
        self.sinmod_data_dict["x"] = x
        self.sinmod_data_dict["y"] = y
        self.sinmod_data_dict["points_xy"] = np.array([x,y])

        self.sinmod_data_dict["dept"] = np.array(self.sinmod_data['depth'])   # Needed?
        #self.sinmod_data_dict["elevation"] = np.array(self.sinmod_data['elevation']) # Needed?

        #self.sinmod_data_dict["u"] = np.array(self.sinmod_data['u_east']) # Needed?
        #self.sinmod_data_dict["v"] = np.array(self.sinmod_data['v_north']) # Needed?

        #self.sinmod_data_dict["us"] = self.sinmod_data_dict["u"].reshape(-1,1) # Needed?
        #self.sinmod_data_dict["vs"] = self.sinmod_data_dict["v"].reshape(-1,1) # Needed?

        #self.sinmod_data_dict["we"] = np.array(self.sinmod_data['w_east']) # Needed?
        #self.sinmod_data_dict["wn"] = np.array(self.sinmod_data['w_north']) # Needed?
        #self.sinmod_data_dict["velocity"] = np.array(self.sinmod_data['w_velocity']) # Needed?

        # loading salinity 
        self.sinmod_data_dict["salinity"] = np.array(self.sinmod_data['salinity'])
        #self.sinmod_data_dict["temperature"] = np.array(self.sinmod_data["temperature"])

        # 
        self.n_timesteps = len(self.sinmod_data_dict["timestamp"])
        self.start_date = self.sinmod_data_dict["timestamp"].units.split(" ")[2]

        # Turning the time into seconds since 1970
        this_date = self.sinmod_data_dict["timestamp"].units.split(" ")[2]
        time_day_start = self.sinmod_data_dict["timestamp"].units.split(" ")[3]

        datetime_str = this_date + " " + time_day_start
        datetime_seconds = datetime.fromisoformat(datetime_str)
        self.start_time_s = datetime_seconds
        self.sinmod_data_dict["time_stamp_s"] = datetime_seconds.timestamp() + self.sinmod_data_dict["time_stamps"]  * 24 * 60 * 60
        

        # Printing loading done
        t2 = time.time()
        print(f"== Loading SINMOD data done == t={t2-t1:,.2f} s")

    def __make_interpolation_functions(self):
        """
        This creates the interpolation functions used to get values from the field 
        """

        x = self.sinmod_data_dict["x"]
        y = self.sinmod_data_dict["y"]
        salinity_loc = self.get_salinity_loc(0,1)
        ind_ocean = np.where((salinity_loc > 0))
        
        # THese are the points we want to use
        points = np.array([x[ind_ocean],y[ind_ocean]])
        
        interpolate_functions = []

        # Iterate over all timestamps 
        for i in range(self.n_timesteps):
            
            # Field at timestep i and depth 
            field = self.get_salinity_loc(i,1)[ind_ocean]

            # Here we create the interpolate functions 
            salinity_interpolator = interpolate.CloughTocher2DInterpolator(points.T, field, tol = 0.1)
            interpolate_functions.append(salinity_interpolator)

        return interpolate_functions

    def get_salinity(self) -> np.ndarray:
        return self.sinmod_data_dict["salinity"]

    def get_lats_lons(self):
        return self.sinmod_data_dict["lats"], self.sinmod_data_dict["lons"]

    def get_xy(self) -> np.ndarray:
        return  self.sinmod_data_dict["x"], self.sinmod_data_dict["y"]

    def get_time_steps_seconds(self) -> np.ndarray:
        return self.sinmod_data_dict["time_stamp_s"]

    def get_salinity_loc(self, time_ind: int, depth_ind: int) -> np.ndarray:
        return self.sinmod_data_dict["salinity"][time_ind, depth_ind, :, :].reshape(-1,1)
    
    def get_points_ocean(self) -> np.ndarray:
        x = self.sinmod_data_dict["x"]
        y = self.sinmod_data_dict["y"]
        salinity_loc = self.get_salinity_loc(0,0)
        ind_ocean = np.where((salinity_loc > 0))
        points = np.array([x[ind_ocean],y[ind_ocean]]).T
        return points

    


    def get_salinity_loc_depth_t(self, depth: int, t: float) -> np.ndarray:
        # Returns the salinity field at a specific time "t"
        # This is done by linear interpolation

        time_stamps = self.sinmod_data_dict["time_stamps"]
    
        # Getting k
        k = 0
        for i in range(len(time_stamps) - 1):
            if t >= time_stamps[i] and t < time_stamps[i+1]:
                k = i
        
        # Salinity k
        s_k = self.sinmod_data_dict["salinity"][k, depth, :, :]
        s_kp1 = self.sinmod_data_dict["salinity"][k + 1, depth, :, :]
        
        # Time k
        t_k = time_stamps[k] 
        t_kp1 = time_stamps[k+1]
        
        # Here we interpolate the t
        sal_time_interp = s_k + (s_kp1  - s_k) * (t - t_k) / (t_kp1 - t_k)
        
        return sal_time_interp

    def get_time_ind_below_above_T(self, T: np.ndarray) -> tuple:
        # T should be in seconds since 1970
        # OPTIMISE
        n = len(T)
        ind_below = []
        ind_above = []

        # This is the sinmod timestamps in unix units
        time_stamps_s = self.sinmod_data_dict["time_stamp_s"]
        for t in T:
            # Getting k
            k = 0
            for i in range(len(time_stamps_s) - 1):
                if t >= time_stamps_s[i] and t < time_stamps_s[i+1]:
                    k = i
            ind_below.append(k)
            ind_above.append(k+1)
        
        return ind_below, ind_above

    def get_time_ind_below_above_t(self, t: np.ndarray):
        # T should be in seconds since 1970
        # OPTIMISE
        ind_below = 0
        ind_above = 1

        # TODO: case where T is not inside start and end timesamp_s
        # This is the sinmod timestamps in unix units
        time_stamps_s = self.sinmod_data_dict["time_stamp_s"]
        # Getting k
        k = 0
        for i in range(len(time_stamps_s) - 1):

            if t >= time_stamps_s[i] and t < time_stamps_s[i+1]:
                k = i
                ind_below = k
                ind_above = k+1
        
        return ind_below, ind_above


    def get_salinity_S_T(self, S: np.ndarray, T: np.ndarray) -> np.ndarray:
        # S points in the 2d plane
        # T time stamps 

        ind_below, ind_above = self.get_time_ind_below_above_T(T)

        time_stamps_s = self.sinmod_data_dict["time_stamp_s"]

        n_points = len(T)
        salinity_point = np.zeros(n_points)

        ind_below = np.array(ind_below, dtype=np.int16)

        k = 0
        for i in range(self.n_timesteps - 1):
            ind = np.where((ind_below == i))
            if len(ind[0]) > 0:
                # s is a list of points
                s = S[ind]

                t_b = time_stamps_s[i]
                t_a = time_stamps_s[i+1]

                salinity_points_b = self.interpolation_functions[i](s)
                salinity_points_a = self.interpolation_functions[i+1](s)

                t_weights = (T[ind] - t_b) / (t_a - t_b)
                sal_time_interp = salinity_points_b + (salinity_points_a  - salinity_points_b) * t_weights

                for s in sal_time_interp:
                    salinity_point[k] = s
                    k += 1

        return salinity_point

    def get_gradient_field(self, time_step, depth, delta = 0.0001):

        x = self.sinmod_data_dict["x"]
        y = self.sinmod_data_dict["y"]
        salinity_loc = self.get_salinity_loc(0,0)
        ind_ocean = np.where((salinity_loc > 0))
        points = np.array([x[ind_ocean],y[ind_ocean]]).T

        n = len(points)
        G_vec = np.zeros((n,2))
        dx = np.array((delta,0))
        dy = np.array((0,delta))

        field_function = self.interpolation_functions[time_step]
        

        for i, xy in enumerate(points):

            gx = (field_function(xy + dx) - field_function(xy - dx)) / (2*delta)
            gy = (field_function(xy + dy) - field_function(xy - dy)) / (2*delta)
            G_vec[i,0] = gx
            G_vec[i,1] = gy
        
        return points, G_vec
    

    def get_salinity_field(self, depth, t):
        x = self.sinmod_data_dict["x"]
        y = self.sinmod_data_dict["y"]
        salinity_loc = self.get_salinity_loc_depth_t(depth, t).reshape(-1,1)
        ind_ocean = np.where((salinity_loc > 0))
        points = np.array([x[ind_ocean],y[ind_ocean]]).T
        return points, salinity_loc[ind_ocean]

    
