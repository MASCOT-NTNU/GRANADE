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
from sklearn.linear_model import LinearRegression


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
TAU = 0.27

# Model noise
SIGMA = 2


# AUV specifications
AUV_SPEED = 1.6
SAMPLE_FREQ = 1


# These are important parameters for the experiment
n_directions = 8
max_points = 10000
horizion =  1000
r = 300
n_iterations = 200
init_r = 70
file_num_prior = 6
file_num_true_field = 6
plot_iter = True
time_lag = 0
add_random_field_exp = True
descicion_rule = "top_p_improvement"
dashboard_type = "full"
restart_AUV = True

# Correction
add_correction = True
beta_0 = 4
beta_1 = 0.9





# Load the sinmod field 
cwd = os.getcwd() 
data_path = cwd + "/src/sinmod_files/"


dir_list = os.listdir(data_path)
sinmod_files = []
for file in dir_list:
    if file.split(".")[-1] == "nc":
        sinmod_files.append(file)

print(sinmod_files)

# Setting up the classes for the experiment
sinmod_field_prior = Prior(data_path + sinmod_files[file_num_prior])
sinmod_field = Prior(data_path + sinmod_files[file_num_true_field])
operation_field = Field()
AUV_data = AUVData(sinmod_field_prior,tau=TAU,sampling_speed=SAMPLE_FREQ,auv_speed=AUV_SPEED , temporal_corrolation=True)
des_rule = DescicionRule(descicion_rule)

# Creating the random field 
t1 = time.time()
sigma_ranadom_field = 0.2
if add_random_field_exp == False:
    sigma_ranadom_field = 0.00000001
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

# Finding a random direction to go in
start_x = np.random.uniform(-2500, 2000)
start_y = np.random.uniform(1000, 5000)
start = np.array([start_x, start_y])
while operation_field.is_loc_legal(np.flip(start)) == False:
    start_x = np.random.uniform(-2500, 2000)
    start_y = np.random.uniform(1000, 5000)
    start = np.array([start_x, start_y])

start = np.array([1800, 3000])
a = start
theta =  np.random.rand(1) * np.pi * 2 

b = np.array([a[1] + init_r * np.cos(theta), a[0] + init_r * np.sin(theta)]).ravel()
while operation_field.is_path_legal(np.flip(a),np.flip(b)) == False: 
    theta =  np.random.rand(1) * np.pi * 2 
    b = np.array([a[0] + init_r * np.cos(theta), a[1] + init_r * np.sin(theta)]).ravel()
    print(b)

# Get data from this field 
S, T, salinity = get_data_transact(a, b, experiment_start_t, sinmod_field, time_shift=time_lag ,add_random_field=add_random_field_exp) 

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
                #plt.plot([a[0], b[0]],[a[1], b[1]], c="green")
            else:
                closest_legal_point = np.flip(operation_field.get_closest_intersect_point(np.flip(a),np.flip(b)))
              
                dist_ab = np.linalg.norm(closest_legal_point - a)

                if dist_ab > r:
                    end_points.append(closest_legal_point)

    path_list = np.array(AUV_data.auv_data["path_list"]) 
   
        

    direction_data = AUV_data.predict_directions(end_points) 
    
    # Here we need the descicion rule

    descicion = des_rule.descicion_rule(direction_data, AUV_data.auv_data)

    b = descicion["end_point"] 
    dist_ab = np.linalg.norm(b - a)
    b = a + (b - a) / dist_ab  * min(dist_ab, r)

    descicion["end_point"] = b

    curr_time = AUV_data.auv_data["T"][-1]

    
    

    plotting_time = time.time()
    if plot_iter:
        plotter = PlotttingFunctions(AUV_data, operation_field, sinmod_field, descicion, direction_data)

        if dashboard_type == "full":
            
            time_elapsed = AUV_data.get_time_elapsed()
            # Plotting the iteration 
            n_x, n_y = 4,3
            fig, ax = plt.subplots(n_y,n_x * 3 - 1,figsize=(n_x * (5 + 1),n_y * 5), gridspec_kw={'width_ratios': [20,1,4, 20, 1,4,20,1,4, 20, 1]})
            
            # change the spacing between subplots
            #fig.subplots_adjust(hspace=.5, wspace=.5)

            # Setting the title for the figure
            iteration_str = "iteration: " + str(i) + "\n"
            total_dist_str = "Total distance: " + str(np.round(((i + 1) * r)/1000,2)) + " km \n"
            total_time_str = "Total time: " + str(np.round((time_elapsed)/3600,2)) + " hours \n"
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
            ax[0,0].set_ylabel("north [m]")
            ax[0,0].set_xlabel("east [m]")
            

            plotter.add_colorbar_variance(ax[0,1], fig)

            plotter.axs_add_limits(ax[0,3]) 
            plotter.plot_measured_salinity(ax[0,3])
            plotter.plot_kriege_gradient(ax[0,3])
            plotter.add_operational_limits(ax[0,3])
            plotter.plot_path(ax[0,3])
            plotter.plot_descicion_paths(ax[0,3], direction_data)
            ax[0,3].set_title("Conditional gradient")
            ax[0,3].set_ylabel("north [m]")
            ax[0,3].set_xlabel("east [m]")

            plotter.add_colorbar_gradient(ax[0,4], fig)

          
            #plotter.plot_true_field(ax[0,4],experiment_start_t + time_elapsed,random_field_function)
            plotter.plot_error_field(ax[0,6], experiment_start_t + time_elapsed, random_field_function)
            plotter.plot_path(ax[0,6], max_points=30)
            ax[0,6].set_title("Error field")
            plotter.add_colorbar_error(ax[0,7], fig)
            ax[0,6].set_ylabel("north [m]")
            ax[0,6].set_xlabel("east [m]")

            plotter.plot_predicted_gradient_best_path(ax[0,9])
            ax[0,9].set_title("Predicted gradient best path")
            ax[0,9].set_ylabel("Directional gradient ")



            plotter.plot_salinity_in_memory(ax[1,0])
            plotter.add_operational_limits(ax[1,0])
            ax[1,0].set_title("Salinity in memory # = " + str(len(AUV_data.auv_data["S"])))
            ax[1,0].set_ylabel("north [m]")
            ax[1,0].set_xlabel("east [m]")

            plotter.add_colorbar_salinity(ax[1,1], fig)

            plotter.axs_add_limits(ax[1,3])
            plotter.plot_kriege_salinity(ax[1,3])
            plotter.add_operational_limits(ax[1,3])
            ax[1,3].set_title("Conditional salinity field")
            ax[1,3].set_xlabel("east [m]")
            ax[1,3].set_ylabel("north [m]")

            plotter.add_colorbar_salinity(ax[1,4], fig)

            plotter.scatter_measured_salinity_prior(ax[1,6])
            ax[1,6].set_title("Measured salinity prior")
            ax[1,6].set_ylabel("Measured salinity")
            ax[1,6].set_xlabel("Prior salinity")

            plotter.plot_score_best_path(ax[1,9])
            ax[1,9].set_title("Score best path")

            plotter.plot_estimated_directional_gradient(ax[2,0])
            ax[2,0].set_title("Estimated directional gradient path")
            ax[2,0].set_ylabel("Directional gradient")

            plotter.plot_estimated_salinity(ax[2,3])
            plotter.plot_measured_salinity(ax[2,3])
            plotter.plot_prior_path(ax[2,3])
            ax[2,3].set_title("Estimated salinity")
            ax[2,3].legend()
            ax[2,3].set_ylabel("Salinity")


            plotter.plot_path(ax[2,6], max_points=10)
            plotter.plot_descicion_paths_score_color(ax[2,6])
            plotter.plot_descicion_end_point(ax[2,6])
            plotter.plot_descicion_score(ax[2,6])
            ax[2,6].set_title("Descicion")
            ax[2,6].set_ylabel("north [m]")
            ax[2,6].set_xlabel("east [m]")
            plotter.add_colorbar_score(ax[2,7], fig)


            plotter.plot_score_all_paths(ax[2,9])
            
            # Removing the axis where there is no colorbar
            ax[0,10].axis('off')
            ax[1,7].axis('off')
            ax[1,10].axis('off')
            ax[2,1].axis('off')
            ax[2,4].axis('off')
            ax[2,10].axis('off')         

            # Remove spacing plots 
            for row in range(n_y):
                for col in range(n_x - 1):
                    ax[row,col*3 + 2].axis('off')

            plt.savefig("src/plots/dashboard/dashboard_"+ str(i) + '.png', dpi=150)
            plt.close()

            all_salinity = AUV_data.get_all_salinity()
            all_prior = AUV_data.get_all_prior()
            reg = LinearRegression().fit(all_prior.reshape(-1,1), all_salinity.reshape(-1,1))
            print("coeff ",reg.coef_," intercept " ,reg.intercept_)

        if dashboard_type == "presentation":
            time_elapsed = AUV_data.get_time_elapsed()
            # Plotting the iteration 
            
            fig, ax = plt.subplots(2,4,figsize=(15,15), gridspec_kw={'width_ratios': [20, 1, 20, 1]})

            # Setting the title for the figure
            iteration_str = "iteration: " + str(i) + "\n"
            total_dist_str = "Total distance: " + str(np.round(((i + 1) * r)/1000,2)) + " km \n"
            total_time_str = "Total time: " + str(np.round((curr_time - experiment_start_t)/3600,2)) + " hours \n"
            fig.suptitle(iteration_str + total_dist_str + total_time_str, fontsize=16)

            plotter.plot_true_field(ax[0,0],experiment_start_t + time_elapsed,random_field_function)
            plotter.plot_path(ax[0,0])
            plotter.plot_descicion_paths(ax[0,0], direction_data)
            plotter.add_colorbar_salinity(ax[0,1], fig)
            plotter.add_operational_limits(ax[0,0])
            plotter.plot_descicion_end_point(ax[0,0])
            ax[0,0].legend()
            ax[0,0].set_xlabel("East [m]")
            ax[0,0].set_ylabel("North [m]")
            ax[0,0].set_title("Sinmod Salinity Field")

            #plotter.plot_gradient_sinmod(ax[0,2], experiment_start_t + time_elapsed)
            plotter.plot_kriege_gradient(ax[0,2])
            plotter.plot_path(ax[0,2])
            plotter.plot_descicion_paths(ax[0,2], direction_data)
            plotter.plot_descicion_end_point(ax[0,0])
            plotter.add_colorbar_gradient(ax[0,3], fig)
            plotter.axs_add_limits(ax[0,2])
            ax[0,2].set_xlabel("East [m]")
            ax[0,2 ].set_ylabel("North [m]")
            ax[0,2].legend()
            ax[0,2].set_title("Gradient Salinity")

            plotter.plot_estimated_salinity(ax[1,0])
            plotter.plot_measured_salinity(ax[1,0])
            ax[1,0].set_title("Estimated salinity along path")
            ax[1,0].legend()
            ax[1,0].set_xlabel("Time [s]")
            ax[1,0].set_ylabel("Salinity")

            plotter.plot_estimated_directional_gradient(ax[1,2])
            ax[1,2].set_title("Estimated directional gradient")
            ax[1,2].legend()
            ax[1,2].set_xlabel("Time [s]")
            ax[1,2].set_ylabel("Directional gradient")

            # Turn off some of the axis
            ax[1,1].axis('off')
            ax[1,3].axis('off')
            
            #fig.tight_layout()
            plt.savefig("src/plots/dashboard/dashboard_"+ str(i) + '.png', dpi=150)
            plt.close()

        if dashboard_type == "simple":
            time_elapsed = AUV_data.get_time_elapsed()
            # Plotting the iteration 
            plotter = PlotttingFunctions(AUV_data, operation_field, sinmod_field, descicion)
            fig, ax = plt.subplots(1,3,figsize=(12,10), gridspec_kw={'width_ratios': [20, 1, 1]})


            # Setting the title for the figure
            iteration_str = "iteration: " + str(i) + "\n"
            total_dist_str = "Total distance: " + str(np.round(((i + 1) * r)/1000,2)) + " km \n"
            total_time_str = "Total time: " + str(np.round((curr_time - experiment_start_t)/3600,2)) + " hours \n"
            fig.suptitle(iteration_str + total_dist_str + total_time_str, fontsize=11)


            plotter.plot_true_field(ax[0],experiment_start_t + time_elapsed,random_field_function, alpha=0.3)
            plotter.plot_path(ax[0])
            plotter.plot_direction_gradient_color(ax[0], direction_data)
            plotter.plot_descicion_end_point(ax[0])
            plotter.add_operational_limits(ax[0])

            ax[0].set_xlabel("East [m]")
            ax[0].set_ylabel("North [m]")
            ax[0].set_title("Agent path and descicion")
            ax[0].legend()


            plotter.add_colorbar_gradient(ax[1], fig)
            plotter.add_colorbar_salinity(ax[2], fig)

            plt.savefig("src/plots/dashboard/dashboard_"+ str(i) + '.png', dpi=150)
            plt.close()




    plotting_time = time.time() - plotting_time 

    # Get data from this field 
    time_elapsed = AUV_data.get_time_elapsed()
    S, T, salinity = get_data_transact(a, b ,experiment_start_t + time_elapsed, sinmod_field, time_shift=time_lag, add_random_field=add_random_field_exp) 

    # Add some correction to the salinty
    if add_correction:
        salinity = salinity * beta_1 + beta_0

    ## First go in a random direct
    AUV_data.add_new_datapoints(S,T - diff_time,salinity)

    iter_speed.append(time.time() - t_1 - plotting_time)
    print(i," Time for iteration: ", round(iter_speed[-1], 2))
    if plot_iter:
        print("Time for plotting: ", round(plotting_time, 2))

    if iter_speed[-1] > 5:
        print("Iteration took more than 5 seconds")
        print("Down sampling data")
        AUV_data.down_sample_points()
            
    
    iter_n_points.append(len(AUV_data.auv_data["S"]))
    
    # Randomly restart the AUV
    if restart_AUV:
        if np.random.uniform() < 0.1:
            print("Restarting AUV")

            print("Data in memory before reset: ", AUV_data.get_number_of_points_in_memory())
            # Try to load the data
            AUV_data = AUVData(sinmod_field_prior,tau=TAU,sampling_speed=SAMPLE_FREQ,auv_speed=AUV_SPEED , temporal_corrolation=True)
            print("Data in memory after rest: ", AUV_data.get_number_of_points_in_memory())
            AUV_data.load_most_recent_data()
            print("Data in memory after reloading data: ", AUV_data.get_number_of_points_in_memory())


# Making a video
fileList = []
for i in range(n_iterations):
    fileList.append("src/plots/dashboard/dashboard_" + str(i) + ".png",)


writer = imageio.get_writer("src/plots/videos/dashboard" + '.mp4', fps=2)

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()




    
