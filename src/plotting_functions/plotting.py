import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np  
from scipy import interpolate

from utility_func import *



class PlotttingFunctions:

    def __init__(self,AUVData,FieldOperation,field_SINMOD,
                 slinity_lim_low = 18,
                 salinity_lim_high = 30) -> None:
        self.auv_data = AUVData
        self.field_operation = FieldOperation
        self.field_sinmod = field_SINMOD    

        # Plotting limits
        self.slinity_lim_low = slinity_lim_low
        self.salinity_lim_high = salinity_lim_high
        self.gradient_lim_low = 0
        self.gradient_lim_high = 0.02

        # Grid limits
        self.x_lim_low = -3000
        self.x_lim_high = 2400
        self.y_lim_low = 0
        self.y_lim_high = 5500  


        # Color maps
        self.cmap_salinity = "turbo"
        self.cmap_gradient = "Reds"
        self.cmap_variance = "bwr"

        # Kriege field functions
        self.salinity_kriege_func = None
        self.variance_kriege_func = None


    @staticmethod
    def sliding_average(x, window_size):
        return np.convolve(x, np.ones(window_size), 'valid') / window_size

        
    def plot_path(self, axis):
        # plot the path of the AUV
        path_list = np.array(self.auv_data.get_auv_path_list())
        axis.plot(path_list[:,0],path_list[:,1], color="black", label="AUV path")

    def add_noth_arrow(self, axis ,a=[-2500, 3500]):
        # Add a north arrow
        x, y, arrow_length = 0.1, 0.9, 0.1
        axis.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=axis.transAxes)

    def add_one_kilometer(self, axis, a=[1000, 5000]):
        # Add a kilometer line
        b = [a[0] + 1000, a[1]]
        axis.plot([a[0], b[0]],[a[1], b[1]], linewidth=2, c="blue", marker="|")
        axis.text(a[0] + 500, a[1] - 100, "1 km")

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

    def plot_salinity_in_memory(self, axis):
        # plot the path of the AUV
        S = self.auv_data.get_points_in_memory()
        salinity = self.auv_data.get_salinity_in_memory()
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

        # Check if the kriege field has been calculated or not
        if self.salinity_kriege_func == None:
            self.kriege_field()

        # Get interpolated values
        interplate_points = self.field_sinmod.get_points_ocean()
        interpolate_values = self.salinity_kriege_func(interplate_points)
        # Plot the interpolated values

        # Check if there are any nan values
        if np.count_nonzero(np.isnan(interpolate_values)) == 0:
            axis.scatter(interplate_points[:,0],interplate_points[:,1], c=interpolate_values,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="Kriging field")
        else:

            if np.count_nonzero(np.isnan(interpolate_values)) > 0.5*len(interpolate_values):
                self.kriege_field()
                interpolate_values = self.salinity_kriege_func(interplate_points)

            inds = np.where(~np.isnan(interpolate_values))
            axis.scatter(interplate_points[inds,0],interplate_points[inds,1], c=interpolate_values[inds],vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="Kriging field")


    def plot_kriege_gradient(self, axis, delta=0.0001):

        if self.salinity_kriege_func == None:
            self.kriege_field()

        # Get interpolated values
        interplate_points = self.field_sinmod.get_points_ocean()

        n = len(interplate_points)
        G_vec = np.zeros((n,2))
        dx = np.array((delta,0))
        dy = np.array((0,delta))

        # TODO: This is a very slow way of doing this. Should do it in a vectorized way
        for i, xy in enumerate(interplate_points):

            gx = (self.salinity_kriege_func(xy + dx) - self.salinity_kriege_func(xy - dx)) / (2*delta)
            gy = (self.salinity_kriege_func(xy + dy) - self.salinity_kriege_func(xy - dy)) / (2*delta)
            G_vec[i,0] = gx
            G_vec[i,1] = gy

        # The absolute value of the gradient
        G_abs = np.linalg.norm(G_vec,axis=1)

        # Check if there are any nan values
        if np.count_nonzero(np.isnan(G_abs)) == 0:
            # Plot the interpolated values
            axis.scatter(interplate_points[:,0],interplate_points[:,1], c=G_abs,vmin=self.gradient_lim_low, vmax=self.gradient_lim_high, cmap=self.cmap_gradient, label="Kriging gradient")

        else:
            
            if np.count_nonzero(np.isnan(G_abs)) > 0.5*len(G_abs):
                self.kriege_field()
                for i, xy in enumerate(interplate_points):

                    gx = (self.salinity_kriege_func(xy + dx) - self.salinity_kriege_func(xy - dx)) / (2*delta)
                    gy = (self.salinity_kriege_func(xy + dy) - self.salinity_kriege_func(xy - dy)) / (2*delta)
                    G_vec[i,0] = gx
                    G_vec[i,1] = gy

            # The absolute value of the gradient
            G_abs = np.linalg.norm(G_vec,axis=1)

            inds = np.where(~np.isnan(G_abs))
            axis.scatter(interplate_points[inds,0],interplate_points[inds,1], c=G_abs[inds],vmin=self.gradient_lim_low, vmax=self.gradient_lim_high, cmap=self.cmap_gradient, label="Kriging gradient")

    def plot_kriege_variance(self, axis):

        if self.variance_kriege_func == None:
            self.kriege_field() 
        
        # Get interpolated values
        interplate_points = self.field_sinmod.get_points_ocean()
        interpolate_values = self.variance_kriege_func(interplate_points)

        # Plot the interpolated values
        axis.scatter(interplate_points[:,0],interplate_points[:,1], c=interpolate_values,vmin=0, cmap=self.cmap_variance, label="Kriging variance")


    def plot_prior_field(self, axis):
        current_time = self.auv_data.get_current_time()
        points, field = self.auv_data.get_prior_salinity_field(current_time)
        axis.scatter(points[:,0],points[:,1], c=field,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="Prior field")


    def add_operational_limits(self, axis):
        # plot the operational limits of the AUV
        xb, yb = self.field_operation.get_exterior_border()
        xo, yo = self.field_operation.get_exterior_obstacle()
        axis.plot(yo, xo, label="obstacle", c="orange")
        axis.plot(yb, xb, label="border", c="green")

    
    def plot_estimated_directional_gradient(self, axis):
        T = self.auv_data.get_all_times()
        G =  self.sliding_average(self.auv_data.get_all_gradient() , 10)
        Var_G = self.sliding_average(self.auv_data.get_all_gradient_variance(), 10)
        axis.plot(T[0:len(G)],G, label="Estimated gradient", c="blue")
        axis.fill_between(T[0:len(G)], G- np.sqrt(Var_G)*1.645 , G+np.sqrt(Var_G)*1.645, alpha=0.2, color="blue", label="95% confidence interval")
        

    def plot_estimated_salinity(self, axis):
        T = self.auv_data.get_all_times()
        m = self.auv_data.get_all_estimated_salinity()
        Var_m = self.auv_data.get_all_salinity_variance()
        axis.plot(T,m, label="Estimated salinity", c="red")
        axis.fill_between(T, m- np.sqrt(Var_m)*1.645 , m+np.sqrt(Var_m)*1.645, alpha=0.2, color="red", label="95% confidence interval")

    def plot_measured_salinity(self, axis):
        T = self.auv_data.get_all_times()
        salinity = self.auv_data.get_all_salinity()
        axis.scatter(T,salinity, label="Measured salinity", c="black",alpha=0.1, marker="x", linestyle="None")


    def plot_prior_path(self, axis):
        T = self.auv_data.get_all_times()
        mu = self.auv_data.get_all_prior()
        axis.plot(T,mu, label="Prior path", c="green", linestyle="dashed")


    def add_colorbar_gradient(self, axis, fig):
        cmap = self.cmap_gradient
        norm = mpl.colors.Normalize(vmin=self.gradient_lim_low, vmax=self.gradient_lim_high)
        cbo = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, label='Gradient [psu/m]')

    def add_colorbar_salinity(self, axis, fig):
        cmap = self.cmap_salinity
        norm = mpl.colors.Normalize(vmin=self.slinity_lim_low, vmax=self.salinity_lim_high)
        cbo = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, label='Salinity [psu]')

    def add_colorbar_variance(self, axis, fig):
        cmap = self.cmap_variance
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cbo = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, label='Variance [psu^2]')


    def scatter_estimated_salinity_prior(self, axis):
        mu = self.auv_data.get_all_prior()
        m = self.auv_data.get_all_estimated_salinity()
        k = 1000
        min_salinity = np.nanmin(m)
        max_salinity = np.nanmax(m)
        if len(mu) > k:
            # Randomly select k points
            inds = np.random.choice(len(mu), k, replace=False)
            axis.scatter(mu[inds],m[inds], label="Estimated salinity",alpha=0.2, c="red", marker="o", linestyle="None")
        else:
            axis.scatter(mu,m, label="Estimated salinity",alpha=0.2, c="red", marker="o", linestyle="None")  

        # Plot a line for the corrolation
        axis.plot([min_salinity,max_salinity],[min_salinity,max_salinity], c="black", linestyle="dashed") 


    def scatter_measured_salinity_prior(self, axis):
        mu = self.auv_data.get_all_prior()
        measured_salinity = self.auv_data.get_all_salinity()
        min_salinity = np.nanmin(measured_salinity)
        max_salinity = np.nanmax(measured_salinity)
        k = 1000
        if len(mu) > k:
            # Randomly select k points
            inds = np.random.choice(len(mu), k, replace=False)
            axis.scatter(mu[inds],measured_salinity[inds],alpha=0.2, label="Measured salinity", c="red", marker="o", linestyle="None")
        else:
            axis.scatter(mu,measured_salinity,alpha=0.2, label="Measured salinity", c="red", marker="o", linestyle="None")   

        # Plot a line for the corrolation
        axis.plot([min_salinity,max_salinity],[min_salinity,max_salinity], c="black", linestyle="dashed")


    def plot_true_field(self, axis ,t,random_field_function):
        depth = 1
        points, field = self.field_sinmod.get_salinity_field(depth, t)
        field = field #+ random_field_function(points)
        axis.scatter(points[:,0],points[:,1], c=field,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="True field")

    def plot_prior_field(self, axis):
        t = self.auv_data.get_current_time()
        points, field = self.auv_data.prior_function.get_salinity_field(1, t)
        axis.scatter(points[:,0],points[:,1], c=field,vmin=self.slinity_lim_low, vmax=self.salinity_lim_high, cmap=self.cmap_salinity, label="Prior field")