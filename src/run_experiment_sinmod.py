import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import seaborn as sns
from scipy import interpolate
from scipy.stats import norm
import random
import imageio
import glob
import itertools
import pickle
import os
from line_profiler import LineProfiler
from scipy.spatial import distance_matrix


# Import classes
from field_SINMOD import field_SINMOD
from WGS import WGS
from field_operation import FieldOperation
from AUV_data import AUVData
from DescicionRule import DescicionRule



def get_points(a,b,t_0):

    dist = np.linalg.norm(b - a)
    total_time = dist / AUV_SPEED
    n_points = int(total_time * SAMPLE_FREQ)
    t_end = t_0 + total_time
    
    T = np.linspace(t_0, t_end, n_points)
    S = np.linspace(a, b, n_points)
    
    return S, T

def get_data_transact(a, b, t_0, field_sinmod, add_noise_position=True, add_noise_data=True):
    
    # This gets the data along a transact
    # We can add measurment noise and location noise
    
    # The measurment noise is defined by TAU
    # THe measurment noise and position noise has mean 0 
    
    S , T = get_points(a,b,t_0)

    if add_noise_position:
        S = S + np.random.normal(0,0.2,S.shape) # need to add this again later
    
    n_samples = len(T)

    mean = 0
    noise = np.random.normal(mean, TAU, n_samples)

    X = field_sinmod.get_salinity_S_T(S,T)

    # TODO: Slow loop
    #for i, p in enumerate(points):

    #    measurments[i] = field_function(p)
        
    if add_noise_data:
        X = X + noise
        
    return S, T, X


#Parameters

# Measurment noise
TAU = 0.4

# Model noise
SIGMA = 2


# AUV specifications
AUV_SPEED = 1.6
SAMPLE_FREQ = 1


# These are important parameters for the experiment
n_directions = 8
max_points = 10000
horizion =  500
r = 250
n_iterations = 50
init_r = 70
file_num = 6



# Load the sinmod field 
cwd = os.getcwd() 
data_path = cwd + "/data_SINMOD/"


dir_list = os.listdir(data_path)
sinmod_files = []
for file in dir_list:
    if file.split(".")[-1] == "nc":
        sinmod_files.append(file)

print(sinmod_files)


sinmod_field = field_SINMOD(data_path + sinmod_files[file_num])
operation_field = FieldOperation()
AUV_data = AUVData(sinmod_field)
des_rule = DescicionRule("top_p_improvement")

    
## This function runs an experiment with the specifications descided
## by the parameters. 

# Keeping time of the experiment
experiment_start_t = sinmod_field.get_time_steps_seconds()[0]



direction_data_list = []
descicions = []


## The parameters for this replicate
# Staring location for the experiment
start = np.array([1000, 3000])
    
# Finding a random direction to go in
a = start
theta =  np.random.rand(1) * np.pi * 2 

b = np.array([a[1] + init_r * np.cos(theta), a[0] + init_r * np.sin(theta)]).ravel()
while operation_field.is_path_legal(np.flip(a),np.flip(b)) == False: 
    theta =  np.random.rand(1) * np.pi * 2 
    b = np.array([a[0] + init_r * np.cos(theta), a[1] + init_r * np.sin(theta)]).ravel()
    print(b)

# Get data from this field 
S, T, salinity = get_data_transact(a,b,experiment_start_t, sinmod_field) 

## First go in a random direct
AUV_data.add_new_datapoints(S,T,salinity)


# Iteration speed
iter_speed = []
iter_n_points = []


# Here is the the loop for the experiment
for i in range(n_iterations):
    t_1 = time.time()

    a = AUV_data.auv_data["S"][-1]
    
    # TODO: Add a no U-turn capacity 
    directions = np.linspace(0,2 * np.pi, n_directions + 1) + np.random.rand(1) * np.pi 
    end_points = []
    for theta in directions:
        b = np.array([a[0] + horizion * np.cos(theta), a[1] + horizion * np.sin(theta)]).ravel()   

        a_prev = AUV_data.auv_data["path_list"][-2]


        u = b - a
        v = a_prev - a
        c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
        angle = np.arccos(np.clip(c, -1, 1)) # if you really want the angle

        # Nu U-turn, the higher the number the higher the movement freedom 
        if angle  > np.pi/4:

            if operation_field.is_path_legal(np.flip(a),np.flip(b)):
                end_points.append(b)
                plt.plot([a[0], b[0]],[a[1], b[1]], c="green")
            else:
                closest_legal_point = np.flip(operation_field.get_closest_intersect_points(np.flip(a),np.flip(b)))
                # TODO add the endpoint in a safe way
                #end_points.append(closest_legal_point)
                plt.plot([a[0], b[0]],[a[1], b[1]], c="red")
                plt.plot([a[0], closest_legal_point[0]],[a[1], closest_legal_point[1]], c="green")


                dist_ab = np.linalg.norm(closest_legal_point - a)

                if dist_ab > r:
                    end_points.append(closest_legal_point)

    path_list = np.array(AUV_data.auv_data["path_list"]) 
    plt.plot(path_list[:,0],path_list[:,1], c="black")       

    operation_field.plot_operational_area(False)
    plt.savefig("src/plots/path_and_tree"+ str(i) + '.png', dpi=150)
    plt.close()  
        

    direction_data = AUV_data.predict_directions(end_points) # This is 99% of the running time
    
    # Here we need the descicion rule

    descicion = des_rule.descicion_rule(direction_data, AUV_data.auv_data)

    b = descicion["end_point"] 
    dist_ab = np.linalg.norm(b - a)
    b = a + (b - a) / dist_ab  * min(dist_ab, r)

    curr_time = AUV_data.auv_data["T"][-1]

    # Get data from this field 
    S, T, salinity = get_data_transact(a,b,curr_time, sinmod_field) 

    ## First go in a random direct
    AUV_data.add_new_datapoints(S,T,salinity)
    
    print(i," Time for iteration: ", round(time.time() - t_1, 2))
            
    iter_speed.append(time.time() - t_1)
    iter_n_points.append(len(AUV_data.auv_data["S"]))

print(iter_speed)

AUV_data.print_auv_data_shape()

plt.show()


S = AUV_data.auv_data["S"]
salinity = AUV_data.auv_data["salinity"]
salinity_loc = sinmod_field.get_salinity_loc(0,0)
ind_ocean = np.where((salinity_loc > 0))
x,y = sinmod_field.get_xy()

print(salinity)
plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=30, alpha=0.05)
plt.scatter(S[:,0], S[:,1], c=salinity, vmin=0, vmax=30)
operation_field.plot_operational_area()
plt.show()
    

plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=30, alpha=0.05)
plt.scatter(S[:,0], S[:,1], c=AUV_data.auv_data["T"])
operation_field.plot_operational_area()
plt.show()

AUV_data.print_timing()
AUV_data.plot_timing()

t_ind, _ = sinmod_field.get_time_ind_below_above(AUV_data.auv_data["T"])
t_ind_np = np.array(t_ind)
m = 0

k = len(sinmod_field.get_time_steps_seconds())
for i in range(k):
    curr_ind = np.where((t_ind_np <= i))

    if len(curr_ind) > 0 and (i in t_ind):


        points, G_vec = sinmod_field.get_gradient_field(i,0)
        G_abs = np.linalg.norm(G_vec,axis=1)
        plt.scatter(points[:,0],points[:,1], c=G_abs, vmin=0, vmax=0.05, cmap="Reds")
        plt.scatter(S[:,0][curr_ind], S[:,1][curr_ind], c=AUV_data.auv_data["m"][curr_ind])
        operation_field.plot_operational_area(False)
        plt.legend()

        plt.savefig("src/plots/gradient_path"+ str(i) + '.png', dpi=150)
        plt.show()




    
