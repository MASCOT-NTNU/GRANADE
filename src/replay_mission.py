# This is a function that replays a mission from the past.

import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os   
from sklearn.linear_model import LinearRegression

# Import classes
from Field import Field
from WGS import WGS
from Prior import Prior
from AUV_data import AUVData
from DescicionRule import DescicionRule
from plotting_functions.plotting import PlotttingFunctions



path_to_data = os.getcwd() + "/src/mission_data/"
expermiment_id = "test_A"
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

dashboard_type = "full"

prior_name = parameters["prior_path"].split("/")[-1]
print(prior_name)
# Getting all the files in the ecperiment folder
files = os.listdir(path_to_experiment)
print(files)

prior = Prior("src/sinmod_files/" + prior_name)
operation_field = Field()

# finding n by checking the largest end number in the files

for i in range(1,20):

    # Load the descicion
    with open(path_to_experiment + '/descicion_data_' + str(i), 'rb') as f:
        descicion = pickle.load(f)

    # load the direction data
    with open(path_to_experiment + '/direction_data_' + str(i), 'rb') as f:
        direction_data = pickle.load(f)

    # load the AUV data
    with open(path_to_experiment + '/all_auv_data_' + str(i), 'rb') as f:
        all_auv_data= pickle.load(f)

    with open(path_to_experiment + '/auv_data_' + str(i), 'rb') as f:
        auv_data= pickle.load(f)
    
    AUV_data = AUVData(prior,
                       tau = tau,
                       sigma = sigma,
                       phi_t = phi_t,
                       phi_d = phi_d,)
    AUV_data.load_from_dict(all_auv_data, auv_data)

    plotter = PlotttingFunctions(AUV_data, operation_field, prior, descicion, direction_data)

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
        #plotter.plot_error_field(ax[0,6], experiment_start_t + time_elapsed, random_field_function)
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
        print("[INFO] coeff ",reg.coef_," intercept " ,reg.intercept_)
