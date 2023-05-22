import matplotlib.pyplot as plt
import numpy as np  
from scipy import interpolate



class PlotttingFunctions:

    def __init__(self,AUVData,FieldOperation,field_SINMOD,
                 slinity_lim_low = 0,
                 salinity_lim_high = 30) -> None:
        self.auv_data = AUVData
        self.field_operation = FieldOperation
        self.field_sinmod = field_SINMOD    

        # Plotting limits
        self.slinity_lim_low = slinity_lim_low
        self.salinity_lim_high = salinity_lim_high
        self.gradient_lim_low = 0
        self.gradient_lim_high = 0.05

        self.x_lim_low = -3000
        self.x_lim_high = 2400
        self.y_lim_low = 0
        self.y_lim_high = 5500  


        # Color maps
        self.cmap_salinity = "viridis"
        self.cmap_gradient = "Reds"
        self.cmap_variance = "bwr"

        # Kriege field functions
        self.salinity_kriege_func = None
        self.variance_kriege_func = None

        
    def plot_path(self, axis):
        # plot the path of the AUV
        path_list = np.array(self.auv_data.get_auv_path_list())
        axis.plot(path_list[:,0],path_list[:,1], color="black", label="AUV path")

    def axs_add_limits(self,axis):
        axis.set_xlim(self.x_lim_low,self.x_lim_high)
        axis.set_ylim(self.y_lim_low,self.y_lim_high)

    def plot_descicion_paths(self, axis, direction_data):
        end_points = direction_data["end_points"]
        start_point = self.auv_data.get_auv_path_list()[-1]
        for p in end_points:
            axis.plot([start_point[0],p[0]],[start_point[1],p[1]], color="Green", label="Descicion path")

    def plot_measured_salinity(self, axis):
        # plot the path of the AUV
        S = self.auv_data.get_all_points()
        salinity = self.auv_data.get_auv_all_salinity()
        axis.scatter(S[:,0],S[:,1], c=salinity,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="AUV path")

    def plot_gradient_sinmod(self, axis, time_step):
        # plot the gradient of the SINMOD field
        points, G_vec = self.field_sinmod.get_gradient_field(time_step,0)
        G_abs = np.linalg.norm(G_vec,axis=1)
        axis.scatter(points[:,0],points[:,1], c=G_abs, vmin=self.gradient_lim_low, vmax=self.gradient_lim_high, cmap=self.cmap_gradient, label="Gradient field")

    
    def kriege_field(self):
        # sample n values from a numpy array
        predict_points = self.field_sinmod.get_points_ocean()
        inds = np.random.choice(np.arange(len(predict_points)), size=4000, replace=False)
        predict_points = predict_points[inds]

        T = np.repeat(self.auv_data.auv_data["T"][-1], len(predict_points))
        mu_pred, sigma_pred , _, _ = self.auv_data.predict_points(predict_points, T)

        # create an interpolator function
        self.salinity_kriege_func = interpolate.CloughTocher2DInterpolator(predict_points, mu_pred)

        # Create an interpolator function for the variance
        self.variance_kriege_func = interpolate.CloughTocher2DInterpolator(predict_points, np.diag(sigma_pred))



    def plot_kriege_salinity(self, axis):

        if self.salinity_kriege_func == None:
            self.kriege_field()

        # Get interpolated values
        interplate_points = self.field_sinmod.get_points_ocean()
        interpolate_values = self.salinity_kriege_func(interplate_points)
        # Plot the interpolated values
        axis.scatter(interplate_points[:,0],interplate_points[:,1], c=interpolate_values,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="Kriging field")

    def plot_kriege_variance(self, axis):

        if self.variance_kriege_func == None:
            self.kriege_field() 
        
        # Get interpolated values
        interplate_points = self.field_sinmod.get_points_ocean()
        interpolate_values = self.variance_kriege_func(interplate_points)

        # Plot the interpolated values
        axis.scatter(interplate_points[:,0],interplate_points[:,1], c=interpolate_values,vmin=0, cmap=self.cmap_variance, label="Kriging variance")





    def add_operational_limits(self, axis):
        # plot the operational limits of the AUV
        self.field_operation.add_plot_operational_area(axis)

