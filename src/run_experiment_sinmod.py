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
from Field import Field
from WGS import WGS
from Prior import Prior
from AUV_data import AUVData
from DescicionRule import DescicionRule
from plotting_functions.plotting import PlotttingFunctions



def get_points(a,b,t_0):

    dist = np.linalg.norm(b - a) 
    total_time = dist / AUV_SPEED
    n_points = int(total_time * SAMPLE_FREQ)
    t_end = t_0 + total_time
    
    T = np.linspace(t_0, t_end, n_points)
    S = np.linspace(a, b, n_points)
    
    return S, T



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
horizion =  800
r = 250
n_iterations = 250
init_r = 70
file_num_prior = 2
file_num_true_field = 7
plot_iter = True
add_random_field = True





# Load the sinmod field 
cwd = os.getcwd() 
data_path = cwd + "/src/sinmod_files/"


dir_list = os.listdir(data_path)
sinmod_files = []
for file in dir_list:
    if file.split(".")[-1] == "nc":
        sinmod_files.append(file)

print(sinmod_files)


sinmod_field_prior = Prior(data_path + sinmod_files[file_num_prior])
sinmod_field = Prior(data_path + sinmod_files[file_num_true_field])
operation_field = Field()
AUV_data = AUVData(sinmod_field_prior, temporal_corrolation=True)
des_rule = DescicionRule("top_p_improvement")

# Creating the random field 
t1 = time.time()
sigma_ranadom_field = 0.2
field_points = sinmod_field.get_points_ocean()
inds = np.random.choice(np.arange(len(field_points)), size=4000, replace=False)
field_points = field_points[inds]
covariance_matrix = AUV_data.make_covariance_matrix(field_points) + np.eye(len(field_points)) * 0.002 #Add something to make the matrix positive definite
print("covariance_matrix", covariance_matrix.shape)
print("field_points", field_points.shape)
L = np.linalg.cholesky(covariance_matrix)
sample = np.random.normal(0,sigma_ranadom_field,len(field_points))
random_field = L @ sample
random_field_function = interpolate.CloughTocher2DInterpolator(field_points, random_field, tol = 0.1)
t2 = time.time()
print("== Time to create random field ==", t2 - t1)
field_points = sinmod_field.get_points_ocean()
plt.scatter(field_points[:,0], field_points[:,1], c=random_field_function(field_points))
plt.colorbar()
plt.close()





def get_data_transact(a, b, t_0, field_sinmod, time_shift=0, add_noise_position=True, add_noise_data=True, add_random_field=True):
    
    # This gets the data along a transact
    # We can add measurment noise and location noise
    
    # The measurment noise is defined by TAU
    # THe measurment noise and position noise has mean 0 
    S , T = get_points(a,b,t_0 )

    if add_noise_position:
        S = S + np.random.normal(0,0.4,S.shape) # need to add this again later
    
    n_samples = len(T)

    mean = 0
    noise = np.random.normal(mean, TAU, n_samples)

    X = field_sinmod.get_salinity_S_T(S,T + time_shift)

        
    if add_noise_data:
        X = X + noise

    if add_random_field:
        X = X + random_field_function(S)

    return S, T, X


    
## This function runs an experiment with the specifications descided
## by the parameters. 

# This is the start time for the true field
experiment_start_t = sinmod_field.get_time_steps_seconds()[0]

# This is the start time for the prior field
prior_start_t = AUV_data.prior_function.get_time_steps_seconds()[0]

# This is the time shift between the two fields
diff_time = experiment_start_t - prior_start_t

if np.abs(diff_time) > 1:
    print("There is a time shift between the prior and the true field")


# There is a time shift between the prior and the true field
# this means that the interpolation function returns the wrong value


direction_data_list = []
descicions = []


## The parameters for this replicate
# Staring location for the experiment
start = np.array([1000, 3000])
    
# Finding a random direction to go in
start_x = np.random.uniform(-2500, 2000)
start_y = np.random.uniform(1000, 5000)
start = np.array([start_x, start_y])
while operation_field.is_loc_legal(np.flip(start)) == False:
    start_x = np.random.uniform(-2500, 2000)
    start_y = np.random.uniform(1000, 5000)
    start = np.array([start_x, start_y])
a = start
theta =  np.random.rand(1) * np.pi * 2 

b = np.array([a[1] + init_r * np.cos(theta), a[0] + init_r * np.sin(theta)]).ravel()
while operation_field.is_path_legal(np.flip(a),np.flip(b)) == False: 
    theta =  np.random.rand(1) * np.pi * 2 
    b = np.array([a[0] + init_r * np.cos(theta), a[1] + init_r * np.sin(theta)]).ravel()
    print(b)

# Get data from this field 
S, T, salinity = get_data_transact(a, b, experiment_start_t, sinmod_field, time_shift=1000) 

## First go in a random direct
# Need to change the time to make it correct
AUV_data.add_new_datapoints(S, T - diff_time,salinity)


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
        if angle  > np.pi/6:

            if operation_field.is_path_legal(np.flip(a),np.flip(b)):
                end_points.append(b)
                plt.plot([a[0], b[0]],[a[1], b[1]], c="green")
            else:
                closest_legal_point = np.flip(operation_field.get_closest_intersect_point(np.flip(a),np.flip(b)))
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
    plt.savefig("src/plots/path_and_tree/path_and_tree"+ str(i) + '.png', dpi=150)
    plt.close()  
        

    direction_data = AUV_data.predict_directions(end_points) 
    
    # Here we need the descicion rule

    descicion = des_rule.descicion_rule(direction_data, AUV_data.auv_data)

    b = descicion["end_point"] 
    dist_ab = np.linalg.norm(b - a)
    b = a + (b - a) / dist_ab  * min(dist_ab, r)

    curr_time = AUV_data.auv_data["T"][-1]

    
    

    plotting_time = time.time()
    if plot_iter:
        # Plotting the iteration 
        plotter = PlotttingFunctions(AUV_data, operation_field, sinmod_field)
        fig, ax = plt.subplots(3,6,figsize=(15,15), gridspec_kw={'width_ratios': [20, 1, 20, 1,20,1]})

        # Setting the title for the figure
        iteration_str = "iteration: " + str(i) + "\n"
        total_dist_str = "Total distance: " + str(np.round(((i + 1) * r)/1000,2)) + " km \n"
        total_time_str = "Total time: " + str(np.round((curr_time - experiment_start_t)/3600,2)) + " hours \n"
        fig.suptitle(iteration_str + total_dist_str + total_time_str, fontsize=16)

        plotter.axs_add_limits(ax[0,0])
        plotter.plot_measured_salinity(ax[0,0])
        plotter.plot_kriege_variance(ax[0,0])
        plotter.add_operational_limits(ax[0,0])
        plotter.plot_path(ax[0,0])
        plotter.plot_descicion_paths(ax[0,0], direction_data)
        plotter.add_noth_arrow(ax[0,0])
        plotter.add_one_kilometer(ax[0,0])
        ax[0,0].set_title("Conditional variance")
        

        plotter.add_colorbar_variance(ax[0,1], fig)

        plotter.axs_add_limits(ax[0,2]) 
        plotter.plot_measured_salinity(ax[0,2])
        plotter.plot_kriege_gradient(ax[0,2])
        plotter.add_operational_limits(ax[0,2])
        plotter.plot_path(ax[0,2])
        plotter.plot_descicion_paths(ax[0,2], direction_data)
        ax[0,1].set_title("Conditional gradient")

        plotter.add_colorbar_gradient(ax[0,3], fig)

        time_elapsed = AUV_data.get_time_elapsed()
        plotter.plot_true_field(ax[0,4],experiment_start_t + time_elapsed,random_field_function)
        ax[0,4].set_title("True field")
        plotter.add_colorbar_salinity(ax[0,5], fig)


        plotter.plot_salinity_in_memory(ax[1,0])
        plotter.add_operational_limits(ax[1,0])
        ax[1,0].set_title("Salinity in memory # = " + str(len(AUV_data.auv_data["S"])))

        plotter.add_colorbar_salinity(ax[1,1], fig)

        plotter.axs_add_limits(ax[1,2])
        plotter.plot_kriege_salinity(ax[1,2])
        plotter.add_operational_limits(ax[1,2])
        ax[1,2].set_title("Kriging salinity")

        plotter.add_colorbar_salinity(ax[1,3], fig)

        plotter.scatter_measured_salinity_prior(ax[1,4])
        ax[1,4].set_title("Measured salinity prior")

        plotter.plot_estimated_directional_gradient(ax[2,0])
        ax[2,0].set_title("Estimated directional gradient")

        
        plotter.plot_estimated_salinity(ax[2,2])
        plotter.plot_measured_salinity(ax[2,2])
        plotter.plot_prior_path(ax[2,2])
        ax[2,2].set_title("Estimated salinity")
        ax[2,2].legend()

        plotter.scatter_estimated_salinity_prior(ax[2,4])
        ax[2,4].set_title("Estimated salinity prior")
        ax[2,4].legend()


        plt.savefig("src/plots/dashboard/dashboard_"+ str(i) + '.png', dpi=150)
        plt.close()
    plotting_time = time.time() - plotting_time

    # Get data from this field 
    time_elapsed = AUV_data.get_time_elapsed()
    S, T, salinity = get_data_transact(a, b ,experiment_start_t + time_elapsed, sinmod_field, time_shift=1000) 

    ## First go in a random direct
    AUV_data.add_new_datapoints(S,T - diff_time,salinity)

    iter_speed.append(time.time() - t_1 - plotting_time)
    print(i," Time for iteration: ", round(iter_speed[-1], 2))
    if plot_iter:
        print("Time for plotting: ", round(plotting_time, 2))

    if iter_speed[-1] > 5:
        AUV_data.down_sample_points()
            
    
    iter_n_points.append(len(AUV_data.auv_data["S"]))




        
print(iter_speed)

AUV_data.print_auv_data_shape()

plt.show()


S = AUV_data.all_auv_data["S"]
salinity = AUV_data.all_auv_data["salinity"]
salinity_loc = sinmod_field.get_salinity_loc(0,0)
ind_ocean = np.where((salinity_loc > 0))
x,y = sinmod_field.get_xy()

print(salinity)
plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=30, alpha=0.05)
plt.scatter(S[:,0], S[:,1], c=salinity, vmin=0, vmax=30)
operation_field.plot_operational_area()
plt.show(block=False)
    

plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=30, alpha=0.05)
plt.scatter(S[:,0], S[:,1], c=AUV_data.all_auv_data["T"])
operation_field.plot_operational_area()
plt.show()

AUV_data.print_timing()
AUV_data.plot_timing()

t_ind, _ = sinmod_field.get_time_ind_below_above(AUV_data.all_auv_data["T"])
t_ind_np = np.array(t_ind)
m = 0

k = len(sinmod_field.get_time_steps_seconds())
for i in range(k):
    curr_ind = np.where((t_ind_np <= i))

    if len(curr_ind) > 0 and (i in t_ind):


        points, G_vec = sinmod_field.get_gradient_field(i,0)
        G_abs = np.linalg.norm(G_vec,axis=1)
        plt.scatter(points[:,0],points[:,1], c=G_abs, vmin=0, vmax=0.05, cmap="Reds")
        plt.scatter(S[:,0][curr_ind], S[:,1][curr_ind], c=AUV_data.all_auv_data["m"][curr_ind])
        operation_field.plot_operational_area(False)
        plt.legend()

        plt.savefig("src/plots/gradient_path/gradient_path"+ str(i) + '.png', dpi=150)
        plt.close()


predict_points = sinmod_field.get_points_ocean()
print(predict_points.shape)

# sample n values from a numpy array
inds = np.random.choice(np.arange(len(predict_points)), size=4000, replace=False)
predict_points = predict_points[inds]

T = np.repeat(AUV_data.auv_data["T"][-1], len(predict_points))
mu_pred, sigma_pred , _, _ = AUV_data.predict_points(predict_points, T)

plt.scatter(predict_points[:,0],predict_points[:,1], c=mu_pred, vmin=0, vmax=30)
plt.colorbar()
plt.show()

plt.scatter(predict_points[:,0],predict_points[:,1], c=np.diag(sigma_pred), vmin=0, vmax=4)
plt.colorbar()
plt.show()



    
