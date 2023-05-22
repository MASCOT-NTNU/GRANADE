
from cProfile import label
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import os
import time

# TODO:
# add a step index list
# add a way to reduce the number of points based on speed



class AUVData:

    def __init__(self,
                    prior_function, 
                    temporal_corrolation: bool = False,
                    tau: float = 0.4,
                    phi_d: float = 200,
                    phi_t: float = 7200,
                    sigma: float = 2,
                    sampling_speed: float = 1,
                    auv_speed: float = 1.6,
                    timing: bool = True) -> None:

        # Parameters for the spatial corrolation 
        self.tau = tau # Measurment noits
        self.phi_d = phi_d
        self.phi_t = phi_t
        self.sigma = sigma
        self.max_points = 2000

        # AUV parameters
        self.auv_speed = auv_speed # m/s
        self.sampling_speed = sampling_speed # s^-1


        # ? should this be here 
        self.prior_function = prior_function

        # AUV data
        self.auv_data = {"has_points": False}
        self.all_auv_data = {"has_points": False}

        self.timing = timing
        self.auv_data_timing = {}
        self.max_time_pr_loop = 20 # seconds

        # If we want to use temporal corrolation
        self.temporal_corrolation = temporal_corrolation

    
    def get_sigma(self) ->  float:
        return self.sigma
    
    def set_sigma(self, sigma):
        self.sigma = sigma

    def get_phi_d(self) ->  float:
        return self.phi_d
    
    def set_phi_d(self, phi_d):
        self.phi_d = phi_d

    def get_tau(self) ->  float:
        return self.tau
    
    def set_tau(self, tau):
        self.tau = tau

    def get_auv_speed(self) ->  float:
        return self.auv_speed
    
    def set_auv_speed(self, auv_speed):
        self.auv_speed = auv_speed

    def get_sampling_speed(self) ->  float:
        return self.sampling_speed
    
    def set_sampling_speed(self, sampling_speed):
        self.sampling_speed = sampling_speed

    def get_max_points(self) ->  float:
        return self.max_points
    
    def set_max_points(self, max_points):
        self.max_points = max_points


    def get_auv_all_salinity(self) -> np.ndarray:
        return self.all_auv_data["salinity"]  
    
    def get_all_points(self) -> np.ndarray:
        return self.all_auv_data["S"]
    
    def get_last_point(self) -> np.ndarray:
        return self.auv_data["S"][-1]
    
    
    def get_auv_data(self) -> dict:
        return self.auv_data
    
    def get_auv_path_list(self) -> np.ndarray:
        return self.auv_data["path_list"]    


    def cov_distance(self, d) -> np.ndarray:    
        # Returns the spatial corroalation for a distance d 
        return self.sigma**2 * np.exp(-(d / self.phi_d)**2)


    # TODO: add a temporal corrolation function
    def cov_temporal(self, t) -> np.ndarray:
        # Returns the temporal corroalation for a time t 
        return np.exp(-(t / self.phi_t)**2)
    
    @staticmethod
    def distance_matrix_one_dimension(vec_1, vec_2) -> np.ndarray:
        return distance_matrix(vec_1.reshape(-1,1), vec_2.reshape(-1,1))


    def make_covariance_matrix(self, S: np.ndarray, T = np.empty((1))) -> np.ndarray:
        D_matrix = distance_matrix(S,S)
        if self.temporal_corrolation:
            T_matrix = self.distance_matrix_one_dimension(T,T)
            return self.cov_distance(D_matrix) * self.cov_temporal(T_matrix)
        return self.cov_distance(D_matrix) 


    def make_covariance_matrix_2(self, S_1: np.ndarray, S_2: np.ndarray, T_1 = np.empty(1), T_2 = np.empty(1)) -> np.ndarray:
        D_matrix = distance_matrix(S_1,S_2)
        if self.temporal_corrolation:
            T_matrix = self.distance_matrix_one_dimension(T_1,T_2)          
            return self.cov_distance(D_matrix) * self.cov_temporal(T_matrix)
        return self.cov_distance(D_matrix)


    def update_covariance_matrix(self, old_cov_matrix,S_old, S_new,T_old=np.empty(1), T_new=np.empty(1)) -> np.ndarray:
        n = len(S_old)
        m = len(S_new)
        
        new_covariance_matrix = np.zeros((n+m,n+m))
        new_covariance_matrix[0:n,0:n] = old_cov_matrix
        
        covariance_matrix_ab = self.make_covariance_matrix_2(S_old, S_new, T_old, T_new)    
        covariance_matrix_ba = np.transpose(covariance_matrix_ab)
        
        covariance_matrix_bb = self.make_covariance_matrix(S_new, T_new)
        
        new_covariance_matrix[n:(n+m),0:n] = covariance_matrix_ba
        new_covariance_matrix[0:n,n:(n+m)] = covariance_matrix_ab
        
        new_covariance_matrix[(n):(n+m),(n):(n+m)] = covariance_matrix_bb
        
        return new_covariance_matrix


    def inverse_matrix_block_symetric(self, A, B, D, A_inv):
        # inverting a matrix with the block shape 
        # | A   B | 
        # | B^T C |
        # where A^-1 is already calculated

        n = A.shape[0]
        m = D.shape[0]
        inverted_matrix = np.zeros((n+m,n+m))

        U = B.T @ A_inv
        V = U.T 

        S = np.linalg.inv(D - B.T @ A_inv @ B)
        
        V_at_S = V @ S

        inverted_matrix[0:n,0:n] = A_inv + V_at_S @ U
        inverted_matrix[n:(n+m),0:n] = - S @ U
        inverted_matrix[0:n,n:(n+m)] = - V_at_S
        inverted_matrix[(n):(n+m),(n):(n+m)] = S

        return inverted_matrix


    def add_first_points(self, S, T, salinity):


        # Improve the path list
        self.auv_data["path_list"] = []
        self.auv_data["path_list"].append(S[0])
        self.auv_data["path_list"].append(S[-1])
        

        # Add the datapoints 
        self.auv_data["S"] = S
        self.auv_data["T"] = T
        self.auv_data["salinity"] = salinity
        self.auv_data["used_points"] = np.repeat(True, len(salinity))

       


        # get the prior for the new points
        mu = self.prior_function.get_salinity_S_T(S, T)
        Sigma = self.make_covariance_matrix(S, T)

        # Gets the conditonal mean and variance
        inv_matrix = np.linalg.inv(Sigma + np.eye(len(mu)) * self.tau**2) 
        self.auv_data["inv_matrix"] = inv_matrix
        self.auv_data["inv_matrix_alt"] = inv_matrix # Remove
        inv_matrix_2 = Sigma @ inv_matrix
        m = mu + inv_matrix_2 @ (salinity - mu)
        Psi = Sigma - inv_matrix_2 @ Sigma
        
        # Store the values
        self.auv_data["Sigma"] =  Sigma
        self.auv_data["mu"] = mu
        self.auv_data["m"] = m
        self.auv_data["Psi"] = Psi


        # Change notes so we know that we have points
        self.auv_data["has_points"] = True


    def add_new_datapoints(self, S_new,T_new, salinity_new):

        start = time.time()
        n_new = len(salinity_new)

        # This function adds the new datapoints and calculates
        # - mu
        # - Sigma
        # - m
        # - Psi
        # - inv_matrix
        # - G
        # - Var_G


        if self.auv_data["has_points"] == False:
            self.add_first_points(S_new, T_new, salinity_new)
        
        else:
            
            # Add the last point to the path list 
            self.auv_data["path_list"].append(S_new[-1])

            # Get previous data
            mu = self.auv_data["mu"]
            S = self.auv_data["S"]
            Sigma = self.auv_data["Sigma"]
            salinity = self.auv_data["salinity"]
            T = self.auv_data["T"]

            # These points are not jet used 
            used_points_new = np.repeat(True, len(salinity_new))
            
            
            # get the prior for the new points
            mu_new = self.prior_function.get_salinity_S_T(S_new, T_new)
            
            # Update the covariance matrix, this saves some time
            n_old, n_new = len(salinity), len(salinity_new)
       
            Sigma_11 = Sigma
            Sigma = self.update_covariance_matrix(Sigma, S, S_new,T, T_new) 

            Sigma_12 = Sigma[0:n_old, n_old:(n_old+n_new)]
            Sigma_22 = Sigma[n_old:(n_old+n_new),n_old:(n_old+n_new)]
            
            # Joining datapoints
            mu = np.concatenate((mu,mu_new))
            salinity = np.concatenate((salinity, salinity_new))
            self.auv_data["S"] = np.concatenate((S,S_new))
            self.auv_data["salinity"] = salinity
            self.auv_data["T"] = np.concatenate((self.auv_data["T"],T_new))
            self.auv_data["used_points"] = np.concatenate((self.auv_data["used_points"],used_points_new))


            # TODO: clean up this 
            # Gets the conditonal mean and variance
            Sigma_11_inv = self.auv_data["inv_matrix"]
            T_1, T_2 = np.eye(n_old) * self.tau**2, np.eye(n_new) * self.tau**2
            inv_matrix = self.inverse_matrix_block_symetric(Sigma_11 + T_1, Sigma_12, Sigma_22 + T_2, Sigma_11_inv)
            inv_matrix_alt = inv_matrix
            #inv_matrix = np.linalg.inv(Sigma + np.eye(n+m) * self.tau**2) 
            inv_matrix2 = Sigma @ inv_matrix
            m = mu + inv_matrix2 @ (salinity - mu)
            Psi = Sigma - inv_matrix2 @ Sigma
            
            #print("error in inversion", np.sum(np.abs(inv_matrix - inv_matrix_alt))) # REMOVE
            #print("Number of nan in iverse alt", np.count_nonzero(np.isnan(inv_matrix_alt))) # REMOVE

            # Store the values
            self.auv_data["Sigma"] =  Sigma
            self.auv_data["mu"] = mu
            self.auv_data["m"] = m
            self.auv_data["Psi"] = Psi
            self.auv_data["inv_matrix"] = inv_matrix
            self.auv_data["inv_matrix_alt"] = inv_matrix_alt # Remove
        
        # Based on this we estimate the gradient
        self.auv_data["G"], self.auv_data["Var_G"] = self.get_gradient(self.auv_data["m"], self.auv_data["S"], self.auv_data["Psi"])

        # Update all data
        self.update_all_data(n_new)


        end = time.time()
        # Store timing
        if self.timing: 
            func_name = "add_new_datapoints"
            if func_name in self.auv_data_timing.keys():
                self.auv_data_timing[func_name]["counter"] += 1 
                self.auv_data_timing[func_name]["total_time"] += end - start
                self.auv_data_timing[func_name]["time_list"].append( end - start)

            else:
                self.auv_data_timing[func_name] = {}
                self.auv_data_timing[func_name]["counter"] = 1 
                self.auv_data_timing[func_name]["total_time"] = end - start
                self.auv_data_timing[func_name]["time_list"] = [end - start]


    def update_all_data(self, n_new: int):

        if self.all_auv_data["has_points"] == False:
            # Add the data to the all_auv_data

            # Vectors 
            self.all_auv_data["S"] = self.auv_data["S"]
            self.all_auv_data["T"] = self.auv_data["T"]
            self.all_auv_data["salinity"] = self.auv_data["salinity"]
            self.all_auv_data["mu"] = self.auv_data["mu"]
            self.all_auv_data["m"] = self.auv_data["m"] 
            self.all_auv_data["G"] = self.auv_data["G"]
            self.all_auv_data["Var_G"] = self.auv_data["Var_G"]

            # Diagonal of the matrices
            self.all_auv_data["dPsi"] = np.diag(self.auv_data["Psi"])
            self.all_auv_data["dSigma"] = np.diag(self.auv_data["Sigma"])


            self.all_auv_data["has_points"] = True
        
        else:
             # Add the data to the all_auv_data

            # Vectors 
            self.all_auv_data["S"] = np.concatenate((self.all_auv_data["S"], self.auv_data["S"][-n_new:]))
            self.all_auv_data["T"] = np.concatenate((self.all_auv_data["T"], self.auv_data["T"][-n_new:]))
            self.all_auv_data["salinity"] = np.concatenate((self.all_auv_data["salinity"], self.auv_data["salinity"][-n_new:]))
            self.all_auv_data["mu"] = np.concatenate((self.all_auv_data["mu"], self.auv_data["mu"][-n_new:]))
            self.all_auv_data["m"] = np.concatenate((self.all_auv_data["m"], self.auv_data["m"][-n_new:]))
            self.all_auv_data["G"] = np.concatenate((self.all_auv_data["G"], self.auv_data["G"][-n_new:]))
            self.all_auv_data["Var_G"] = np.concatenate((self.all_auv_data["Var_G"], self.auv_data["Var_G"][-n_new:]))

            # Diagonal of the matrices
            self.all_auv_data["dPsi"] = np.concatenate((self.all_auv_data["dPsi"], np.diag(self.auv_data["Psi"])[-n_new:]))
            self.all_auv_data["dSigma"] = np.concatenate((self.all_auv_data["dSigma"], np.diag(self.auv_data["Sigma"])[-n_new:]))





    def down_sample_points(self):
        # This function removes half of the points in the data
        # This is done to save time
        old_data = self.auv_data
        new_data = {}

        ind = [True if i % 2 == 0 else False for i in range(len(old_data["S"]))]
        new_data["S"] = old_data["S"][ind]
        new_data["T"] = old_data["T"][ind]
        new_data["salinity"] = old_data["salinity"][ind]
        new_data["Sigma"] = old_data["Sigma"][ind][:,ind]
        new_data["mu"] = old_data["mu"][ind]
        new_data["m"] = old_data["m"][ind]
        new_data["Psi"] = old_data["Psi"][ind][:,ind]
        new_data["G"] = old_data["G"][ind[0:-1]]
        new_data["Var_G"] = old_data["Var_G"][ind[0:-1]]
        new_data["inv_matrix"] =  np.linalg.inv(new_data["Sigma"] + np.eye(len(new_data["mu"])) * self.tau**2)

        # Adding the unchanged data
        new_data["has_points"] = old_data["has_points"]
        new_data["used_points"] = old_data["used_points"]
        new_data["path_list"] = old_data["path_list"]
        
        # Store the down sampled data
        self.auv_data = new_data




        


        

    def predict_points(self, P_predict: np.ndarray, T_predict):

        if self.auv_data["has_points"] == False:

            # If we have no measurments then we can only give the unconditional values

            # the prior
            mu_predict = self.prior_function.get_salinity_S_T(P_predict, T_predict)

            # Sigma 
            Sigma_PP = self.make_covariance_matrix(P_predict, T_predict)

            # Estimate the gradient
            G, Var_G = self.get_gradient(mu_predict, P_predict, Sigma_PP)

            return mu_predict, Sigma_PP, G, Var_G

        else:
            S = self.auv_data["S"]
            T = self.auv_data["T"]
            Sigma_S = self.auv_data["Sigma"]
        
            Sigma_SP = self.update_covariance_matrix(Sigma_S, S, P_predict,T, T_predict)
            
            n = len(S)
            m = len(P_predict)
            

            cov_sy = Sigma_SP[0:n, n:(n+m)]
            cov_yy = Sigma_S
            cov_ss = Sigma_SP[n:(n+m),n:(n+m)]
            
            
            M = np.transpose(cov_sy) @ self.auv_data["inv_matrix"]
            
            # the prior
            mu_S = self.auv_data["mu"]

            salinity_S = self.auv_data["salinity"]

            # get the prior for the new points
            mu_predict = self.prior_function.get_salinity_S_T(P_predict, T_predict)
            
            mu_predict = mu_predict +  M @ (salinity_S  - mu_S)
            Psi_predict = cov_ss - M @ cov_sy


            # Estimate the gradient
            G, Var_G = self.get_gradient(mu_predict, P_predict, Psi_predict)
            
            return mu_predict, Psi_predict, G, Var_G


    def get_conditional_mean_variance(self):
        return self.auv_data["m"], self.auv_data["Psi"]

    def get_unconditional_mean_variance(self):
        return self.auv_data["mu"], self.auv_data["Sigma"]

    def get_gradient_variance(self):
        return self.auv_data["G"], self.auv_data["Var_G"]



    def print_auv_data_shape(self):
        if self.auv_data["has_points"] == True:

            for key in self.auv_data.keys():
                print(key ,end=" ")

                if isinstance(self.auv_data[key], np.ndarray):
                    print(self.auv_data[key].shape)
                else:
                    print("")



    @staticmethod
    def get_gradient(m, S, Psi):
        
        # m - the expected salinity
        # Psi - is the estimated conditional covariance matrix
        # S - the location where the data points was measured

        diff = m[1:] - m[:-1]
        dist = np.linalg.norm(S[1:] - S[:-1], axis=1)
        
        # G is the estimated Gradient
        G = diff / dist

        # Add the variance 
        Var_G = np.zeros(len(G))
        for i in range(len(G)):
            Var_G[i] = (Psi[i,i] + Psi[i+1,i+1] - 2 * Psi[i,i+1]) / dist[i]**2

        # Returning the gradient
        if np.min(Var_G) < 0:
            plt.plot(Var_G)
            plt.show()

        if np.count_nonzero(np.isnan(Var_G)):  # Remove
            plt.plot(Var_G)
            plt.show(Var_G)
        return G, Var_G


    def get_points(self, a,b,t_0):
        
        dist = np.linalg.norm(b - a)
        total_time = dist / self.auv_speed
        n_points = int(total_time * self.sampling_speed)
        t_end = t_0 + total_time
        
        T = np.linspace(t_0, t_end, n_points)
        S = np.linspace(a, b, n_points)
        
        return S, T





    def predict_directions(self, endpoints):
    
        # The data we want to store
        direction_data = {}
        
        direction_data["salinity_directions"] = []
        direction_data["points_directions"] = []
        direction_data["gradient_directions"] = []
        direction_data["var_gradient_directions"] = []
        direction_data["end_points"] = []
        #direction_data["true_salinity_direction"] = []
        #direction_data["true_transact"] = []
        
        # The nescisary data
        S = self.auv_data["S"]
        m_k = self.auv_data["m"]
        Sigma = self.auv_data["Sigma"]
        salinity = self.auv_data["salinity"]
        a = S[-1]
        t_0 = self.auv_data["T"][-1]
        
        # Iterate over all the directions
        # The directions is a list of angles
        #inv_matrix = self.auv_data["inv_matrix"]
        inv_matrix = self.auv_data["inv_matrix"]
        
        for j in range(len(endpoints)):
                
            # The current angle 
            b = endpoints[j]

            # These are the points we want to predict for
            P_predict, T_predict = self.get_points(a,b,t_0)
            
            if len(T_predict) > 2:
                
                # Predicting the transact  
                G, Var_G, transact_points, transact_salinity = self.estimate_gradient_transact(inv_matrix, a, b, t_0)
        

                # Getting the true gradient, this is not available in real life
                # TODO: some bad names here
                #transact, salinity_transact = get_data_transact(a,b , add_noise_position=False, add_noise_data=False)


                # Save the data 
                #  From prediction
                direction_data["gradient_directions"].append(G)
                direction_data["var_gradient_directions"].append(Var_G)
                direction_data["end_points"].append(b)
                direction_data["points_directions"].append(transact_points)
                direction_data["salinity_directions"].append(transact_salinity)

                #  Data from the true field 
                #direction_data["true_salinity_direction"].append(salinity_transact)
                #direction_data["true_transact"].append(transact)
        
        # The number of directions used
        direction_data["n_directions"] = len(direction_data["salinity_directions"])
        
        return direction_data

    
    def estimate_gradient_transact(self, inv_matrix, a, b, t_0):
        start = time.time()

        # a - transact start
        # b - transact end
        
        # mu and cov are estimated from the data
        # data_points are the location where the data is measured
        # data_value is the value for the data, mu is then smooth relative to this
        
        P_predict, T_predict = self.get_points(a,b, t_0)
        
        # old values
        Sigma = self.auv_data["Sigma"]
        salinity_S = self.auv_data["salinity"]
        S = self.auv_data["S"]
        T = self.auv_data["T"]  
        mu_S = self.auv_data["mu"]

        # This is where 70 - 90 % of the time goes
        cov_matrix_large = self.update_covariance_matrix(Sigma, S, P_predict, T , T_predict)
        
        # The size of the different datasets
        n = len(salinity_S)
        m = len(T_predict)
        
        Sigma_SP = cov_matrix_large[0:n, n:(n+m)]
        Sigma_PP = cov_matrix_large[n:(n+m),n:(n+m)]
        
        # getting the prior for the points
        mu_P = self.prior_function.get_salinity_S_T(P_predict, T_predict)
        
        # Get the conditional 
        M = np.transpose(Sigma_SP) @ inv_matrix
        m_P = mu_P +  M @ (salinity_S - mu_S)
        Psi_PP = Sigma_PP - M @ Sigma_SP
        

        
        G, Var_G = self.get_gradient(m_P, P_predict, Psi_PP)

        end = time.time()
        # Store timing
        if self.timing: 
            func_name = "estimate_gradient_transact"
            if func_name in self.auv_data_timing.keys():
                self.auv_data_timing[func_name]["counter"] += 1 
                self.auv_data_timing[func_name]["total_time"] += end - start
                self.auv_data_timing[func_name]["time_list"].append( end - start)

            else:
                self.auv_data_timing[func_name] = {}
                self.auv_data_timing[func_name]["counter"] = 1 
                self.auv_data_timing[func_name]["total_time"] = end - start
                self.auv_data_timing[func_name]["time_list"] = [end - start]

        return G, Var_G, S, m_P

    def print_timing(self):
        if self.timing:
            for key in self.auv_data_timing.keys():
                print("Function: ", key)
                timing_dat = self.auv_data_timing[key]
                print("total calls: ", round( timing_dat["counter"],2) , " s")
                print("Total time: ", round( timing_dat["total_time"], 2), " s")

    def plot_timing(self):
        if self.timing:
            for key in self.auv_data_timing.keys():
        
                timing_dat = self.auv_data_timing[key]
                x = np.linspace(0,1, len(timing_dat["time_list"]))
                plt.plot(x, timing_dat["time_list"], label = key)

            plt.legend()
            plt.title("time per call for function")
            plt.ylabel("Time seconds")
            plt.show()


            for key in self.auv_data_timing.keys():
        
                timing_dat = self.auv_data_timing[key]
                
                timing_dat = self.auv_data_timing[key]
                x = np.linspace(0,1, len(timing_dat["time_list"]))
                y = np.cumsum(timing_dat["time_list"])
                plt.plot(x, y, label = key)

            plt.legend()
            plt.title("Cumulative time for function")
            plt.ylabel("Time seconds")
            plt.show()


    


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import time


        
    from field_SINMOD import field_SINMOD
    from WGS import WGS
    from field_operation import FieldOperation
    from AUV_data import AUVData   


    # AUV specifications
    AUV_SPEED = 1
    SAMPLE_FREQ = 1

    TAU = 0.4



    # Load the sinmod field 
    cwd = os.getcwd() 
    data_path = cwd + "/src/sinmod_files/"


    dir_list = os.listdir(data_path)
    sinmod_files = []
    for file in dir_list:
        if file.split(".")[-1] == "nc":
            sinmod_files.append(file)

    print(sinmod_files)

    file_num = 0
    sinmod_field = field_SINMOD(data_path + sinmod_files[file_num])
    operation_field = FieldOperation()
    AUV_data = AUVData(sinmod_field,temporal_corrolation=True)


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







    t_0 = sinmod_field.sinmod_data_dict["time_stamp_s"][50] # start time 
    a = np.array([0,2000]) # start point
    b = np.array([1000,3000]) # end point
    c = np.array([2000,3000]) # point 3
    S, T, salinity = get_data_transact(a,b,t_0, sinmod_field)   



    salinity_loc = sinmod_field.get_salinity_loc(70,0)
    ind_ocean = np.where((salinity_loc > 0))
    print(np.nanmax(salinity_loc))
    x,y = sinmod_field.get_xy()

    plt.scatter(x[ind_ocean],y[ind_ocean],c=salinity_loc[ind_ocean], vmin=0, vmax=35)
    plt.scatter(S[:,0], S[:,1], c=salinity, vmin=0, vmax=35)
    plt.show()

   

    AUV_data.add_new_datapoints(S, T, salinity)
    AUV_data.print_auv_data_shape()

    m, psi = AUV_data.get_conditional_mean_variance()

    #plt.plot(sal, c = "Red")
    n = np.arange(len(salinity))
    plt.scatter(n,salinity , c="Green")
    plt.plot(n,m)
    plt.show()

    plt.plot(T)
    plt.show()

   

    t_next = T[-1]
    S, T, salinity = get_data_transact(b,c,t_next, sinmod_field)
    S_true, T_true, salinity_true = get_data_transact(b,c,t_next, sinmod_field, add_noise_data=False, add_noise_position=False)

    print(np.sum(T - T_true))

    m_predic, psi_predict, G_predict, Var_G = AUV_data.predict_points(S_true, T_true)
    m_interpolate = AUV_data.auv_data["m"]
    Psi_interpolate = AUV_data.auv_data["Psi"]
    T_interpolate = AUV_data.auv_data["T"]


    n = np.arange(len(salinity))
    plt.plot(T_true,m_predic , c="Green", label="predicted")
    plt.plot(T_true,m_predic + np.sqrt(np.diag(psi_predict)) * 1.645, c="Green", linestyle="--")
    plt.plot(T_true,m_predic - np.sqrt(np.diag(psi_predict)) * 1.645, c="Green", linestyle="--")
    plt.plot(T_true,salinity_true, c="blue", label="True value")
    plt.plot(T_interpolate, m_interpolate,c="green", label="interpolate")
    plt.plot(T_interpolate, m_interpolate + np.sqrt(np.diag(Psi_interpolate)) * 1.645,c="green" , linestyle="--")
    plt.plot(T_interpolate, m_interpolate - np.sqrt(np.diag(Psi_interpolate)) * 1.645,c="green", linestyle="--")
    plt.legend()
    plt.show()

    AUV_data.add_new_datapoints(S, T, salinity)
    AUV_data.print_auv_data_shape()

    end_points = [np.array([2050,3000]), np.array([2050,3050]), np.array([2000,3050])]
    print(AUV_data.predict_directions(endpoints=end_points))

    AUV_data.print_timing()
    AUV_data.plot_timing()



    AUV_data.down_sample_points()