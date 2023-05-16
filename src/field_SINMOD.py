
import os
import netCDF4
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from datetime import date
from scipy import interpolate



from WGS import WGS

class field_SINMOD:

    def __init__(self, sinmod_path_str) -> None:
        self.sinmod_path = sinmod_path_str

        self.sinmod_data = netCDF4.Dataset(sinmod_path_str)
        self.sinmod_data_dict = {}
        

        self.n_depts: int
        self.n_timesteps: int
        self.start_date: str
        self.start_second: float


        self.load_sinmod_data()

        t1 = time.time()  #REMOVE
        self.interpolation_functions = self.make_interpolation_functions()
        t2 = time.time()  #REMOVE
        print(f"== Create interpolate functions == t={t2-t1:,.2f} s")  #REMOVE
    

    def set_sinmod_path(self, path_string: str) -> None:
        self.sinmod_path = path_string


    def load_sinmod_data(self) -> None:
        t1 = time.time()
        
        # Loading the data into the dict
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

        self.sinmod_data_dict["dept"] = np.array(self.sinmod_data['depth'])
        self.sinmod_data_dict["elevation"] = np.array(self.sinmod_data['elevation'])

        self.sinmod_data_dict["u"] = np.array(self.sinmod_data['u_east'])
        self.sinmod_data_dict["v"] = np.array(self.sinmod_data['v_north'])

        self.sinmod_data_dict["us"] = self.sinmod_data_dict["u"].reshape(-1,1)
        self.sinmod_data_dict["vs"] = self.sinmod_data_dict["v"].reshape(-1,1)

        self.sinmod_data_dict["we"] = np.array(self.sinmod_data['w_east'])
        self.sinmod_data_dict["wn"] = np.array(self.sinmod_data['w_north'])
        self.sinmod_data_dict["velocity"] = np.array(self.sinmod_data['w_velocity'])
        self.sinmod_data_dict["salinity"] = np.array(self.sinmod_data['salinity'])
        self.sinmod_data_dict["temperature"] = np.array(self.sinmod_data["temperature"])

           
        #self.n_depts = 
        self.n_timesteps = len(self.sinmod_data_dict["timestamp"])
        self.start_date = self.sinmod_data_dict["timestamp"].units.split(" ")[2]

        # Turning the time into seconds 
        this_date = self.sinmod_data_dict["timestamp"].units.split(" ")[2]
        time_day_start = self.sinmod_data_dict["timestamp"].units.split(" ")[3]

        datetime_str = this_date + " " + time_day_start
        datetime_seconds = datetime.fromisoformat(datetime_str)
        self.start_second = datetime_seconds
        self.sinmod_data_dict["time_stamp_s"] = datetime_seconds.timestamp() + self.sinmod_data_dict["time_stamps"]  * 24 * 60 * 60
        
        t2 = time.time()
        print(f"== Loading SINMOD data done == t={t2-t1:,.2f} s")

    def make_interpolation_functions(self):
        x = self.sinmod_data_dict["x"]
        y = self.sinmod_data_dict["y"]
        salinity_loc = self.get_salinity_loc(0,0)
        ind_ocean = np.where((salinity_loc > 0))

        points = np.array([x[ind_ocean],y[ind_ocean]])
        
        interpolate_functions = []
        for i in range(self.n_timesteps):
           
            field = self.get_salinity_loc(i,0)[ind_ocean]

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


    def get_salinity_dept_t(self, depth: int, t: float) -> np.ndarray:
        # Returns the salinity field at a specific time
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

    def get_salinity_S_t(self, S: np.ndarray, t: float) -> np.ndarray:
        salinity_loc = self.get_salinity_dept_t(1, t).reshape(-1,1)
        points_xy = self.sinmod_data_dict["points_xy"]
        
        points_xy = points_xy[:,:,0]
        salinity_loc = salinity_loc[:,0]

        ind = np.where((salinity_loc > 0))
        
      
        salinity_interpolator = interpolate.CloughTocher2DInterpolator(points_xy.T[ind], salinity_loc[ind], tol = 0.1)
        return salinity_interpolator(S)



    def get_salinity_s_t(self, s: np.ndarray, t: float) -> np.ndarray:
        salinity_loc = self.get_salinity_dept_t(1, t).reshape(-1,1)
        points_xy = self.sinmod_data_dict["points_xy"]
        
        points_xy = points_xy[:,:,0]
        salinity_loc = salinity_loc[:,0]

        x = points_xy[:,0]
        y = points_xy[:,1]

        ind = np.where((((s[0]-x)**2 + (s[1]-y)**2)**(1/2) < 1000))

        ind = np.where((salinity_loc > 0) * (np.linalg.norm(points_xy.T - s) < 1000))
        
      
        salinity_interpolator = interpolate.CloughTocher2DInterpolator(points_xy.T[ind], salinity_loc[ind], tol = 0.1)
        return salinity_interpolator(S)


    def get_time_ind_below_above(self, T: np.ndarray):
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

        ind_below, ind_above = self.get_time_ind_below_above(T)

        time_stamps_s = self.sinmod_data_dict["time_stamp_s"]

        n_points = len(T)
        salinity_point = np.zeros(n_points)

        ind_below = np.array(ind_below, dtype=np.int16)

        k = 0
        for i in range(self.n_timesteps - 1):
            ind = np.where((ind_below == i))
            if len(ind[0]) > 0:
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



if __name__=="__main__":
    import matplotlib.pyplot as plt
    import time

    
    cwd = os.getcwd() 
    data_path = cwd + "/data_SINMOD/"

    
    dir_list = os.listdir(data_path)
    sinmod_files = []
    for file in dir_list:
        if file.split(".")[-1] == "nc":
            sinmod_files.append(file)

    print(sinmod_files)

    file_num = 0
    sinmod_field = field_SINMOD(data_path + sinmod_files[file_num])


    sampling_frequency = 0.1 # s^-1
    auv_speed = 1.6 # m/2
    t_0 = sinmod_field.sinmod_data_dict["time_stamp_s"][70] # start time 
    a = np.array([0,2000]) # start point
    b = np.array([1000,3000]) # end point

    def get_points(a,b,t_0):
    
        dist = np.linalg.norm(b - a)
        total_time = dist / auv_speed
        n_points = int(total_time * sampling_frequency)
        t_end = t_0 + total_time
        
        T = np.linspace(t_0, t_end, n_points)
        S = np.linspace(a, b, n_points)
        
        return S, T

    S, T = get_points(a,b,t_0)

    t1 = time.time()
    sal_alt = sinmod_field.get_salinity_S_T(S,T)
    t2 = time.time()
    print(len(T), t2-t1)

    t1 = time.time()
    sal_alt = sinmod_field.get_salinity_S_T(S,T)
    t2 = time.time()
    print(len(T), t2-t1)


    #plt.plot(sal, c = "Red")
    plt.plot(sal_alt , c="Green")
    plt.show()

    

    salinity_loc = sinmod_field.get_salinity_loc(70,0)
    ind_ocean = np.where((salinity_loc > 0))
    print(np.nanmax(salinity_loc))
    x,y = sinmod_field.get_xy()

    plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=35)
    plt.scatter(S[:,0], S[:,1], c=sal_alt,cmap="Reds", vmin=0, vmax=35)
    plt.show()


    points, G_vec = sinmod_field.get_gradient_field(1,0)
    G_abs = np.linalg.norm(G_vec,axis=1)
    plt.scatter(points[:,0],points[:,1], c=G_abs, vmin=0, vmax=0.05, cmap="Reds")
    plt.show()






