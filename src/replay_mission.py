# This is a function that replays a mission from the past.

import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os   
import pandas as pd
import imageio
from sklearn.linear_model import LinearRegression

# Import classes
from Field import Field
from WGS import WGS
from Prior import Prior
from AUV_data import AUVData
from DescicionRule import DescicionRule
from plotting_functions.plotting import PlotttingFunctions



path_to_data = os.getcwd() + "/src/mission_data/save_counter_data/"
expermiment_id = "mission_2023.06.22_A"
path_to_experiment = path_to_data + expermiment_id
experiment_start_t = time.time()

# Load parameters
with open(path_to_experiment + '/parameters', 'rb') as f:
    parameters = pickle.load(f)


print(parameters)


phi_t = parameters["phi_t"]
phi_d = parameters["phi_d"]
tau = parameters["tau"]
sigma = parameters["sigma"]
descicion_rule = parameters["descicion_rule"]
r = parameters["radius"]

# Setting up the time difference
high_tide_prior = parameters["high_tide_prior"]
high_tide_today = parameters["high_tide_today"]
time_diff = int((high_tide_today - high_tide_prior).total_seconds())
print("High tide prior: ", high_tide_prior)
print("High tide today: ", high_tide_today)
print("Time diff: ", high_tide_prior - high_tide_today)
print("Time diff (s): ", time_diff)


dashboard_type = "full"

prior_name = parameters["prior_path"].split("/")[-1]
print(prior_name)
# Getting all the files in the ecperiment folder
files = os.listdir(path_to_experiment)
print(files)

prior = Prior("src/sinmod_files/" + prior_name)
operation_field = Field()

# finding n by checking the largest end number in the files

# This is the numer of iterations completed
n = 40
for i in range(1,n):

    # Load the descicion
    with open(path_to_experiment + '/descicion_data_' + str(i), 'rb') as f:
        descicion = pickle.load(f)

    # load the direction data
    with open(path_to_experiment + '/direction_data_' + str(i), 'rb') as f:
        direction_data = pickle.load(f)

    # load the AUV data
    with open(path_to_experiment + '/all_auv_data_' + str(i), 'rb') as f:
        all_auv_data= pickle.load(f)

    # load the AUV data in memory
    with open(path_to_experiment + '/auv_data_' + str(i), 'rb') as f:
        auv_data= pickle.load(f)

    # Here we load all the data
    S = np.array(all_auv_data["S"])
    T = np.array(all_auv_data["T"])
    salinity = np.array(all_auv_data["salinity"])

    ind = np.where(salinity > 15)

    # Filtering out the data
    S = S[ind]
    T = T[ind]
    salinity = salinity[ind]
    
    AUV_data = AUVData(prior,
                       tau = tau,
                       sigma = sigma,
                       phi_t = phi_t,
                       phi_d = phi_d,
                       time_diff_prior=time_diff)
    #AUV_data.add_new_datapoints(S, T, salinity, np.repeat(1,len(T)))
    AUV_data.load_from_dict(all_auv_data, auv_data)

    plotter = PlotttingFunctions(AUV_data, operation_field, prior, descicion, direction_data ,
                                 gradient_lim_high=0.01,
                                 prior_time_correction=-time_diff)

    def random_field_function(x,y):
        return 0 



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

        plotter.axs_add_limits_small(ax[0,0])
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

        plotter.axs_add_limits_small(ax[0,3]) 
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
        plotter.plot_estimated_gradient_path(ax[0,6])
        #plotter.plot_path(ax[0,6], max_points=30)
        ax[0,6].set_title("Gradient path")
        plotter.add_colorbar_gradient(ax[0,7], fig)
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

        plotter.axs_add_limits_small(ax[1,3])
        plotter.plot_kriege_salinity(ax[1,3])
        plotter.plot_path(ax[1,3])
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
        #plotter.plot_prior_path(ax[2,3])
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


        plotter.plot_direction_gradient_color(ax[2,9])
        
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
        print("[INFO] coeff ",reg.coef_," intercept " ,reg.intercept_)

   
    if dashboard_type == "presentation":
        time_elapsed = AUV_data.get_time_elapsed()
        # Plotting the iteration 
        
        fig, ax = plt.subplots(2,4,figsize=(15,15), gridspec_kw={'width_ratios': [20, 1, 20, 1]})

        # Setting the title for the figure
        iteration_str = "iteration: " + str(i) + "\n"
        total_dist_str = "Total distance: " + str(np.round(((i + 1) * r)/1000,2)) + " km \n"
        total_time_str = "Total time: " + str(np.round((time_elapsed)/3600,2)) + " hours \n"
        fig.suptitle(iteration_str + total_dist_str + total_time_str, fontsize=16)

        plotter.plot_prior_field(ax[0,0],experiment_start_t + time_elapsed )
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


# Making a video
fileList = []
for i in range(1,n):
    fileList.append("src/plots/dashboard/dashboard_" + str(i) + ".png",)


writer = imageio.get_writer("src/plots/videos/dashboard" + '.mp4', fps=0.5)

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()



