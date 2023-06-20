
import numpy as np
import os
import time 
import math 
import rospy
import datetime
import pickle

from AUV import AUV
from WGS import WGS
from Field import Field
from Prior import Prior
from AUV_data import AUVData
from DescicionRule import DescicionRule





class Agent:

    def __init__(self,
                 experiment_id = "new",
                 sinmod_file_name = "samples_2022.06.21") -> None:
        """
        Setting up the agent 
        """
        print("[ACTION] Setting up the agent")

        # Thses are the parameters for the mission
        self.wp_start =  np.array([1800, 3000])
        self.n_directions = 8
        self.horizion = 1000  # meters, how far the agent can see
        self.radius = 250 # meters, how far the agent will move 
        self.radius_init = 150 # meters, how far the agent will move on the first iteration
        self.descicion_rule_str = "top_p_improvement"
        self.prior_path = os.getcwd() + "/src/sinmod_files/" + sinmod_file_name + ".nc"
        self.start_from_current_location = True

        # Tide data 
        self.prior_date = datetime.datetime(2022,6,21)            # SET EVERY MISSION
        self.date_today = datetime.datetime(2023,6,20)            # SET EVERY MISSION
        self.high_tide_prior = datetime.datetime(2022,6,21,5,30)  # SET EVERY MISSION
        self.high_tide_today = datetime.datetime(2023,6,20,1,40)  # SET EVERY MISSION
        correction = 0
        self.diff_date_s = int((self.date_today - self.prior_date).total_seconds())
        self.diff_tide_time = int((self.high_tide_today - self.high_tide_prior).total_seconds() - self.diff_date_s)
        self.diff_time_s = int((self.high_tide_today - self.high_tide_prior).total_seconds()) + correction
        print("[INFO] Time <now> in prior time: ", datetime.datetime.fromtimestamp(time.time() - self.diff_time_s))
        print("[NOTE] Check that this seems reasonable, if it is wrong make a correction in the code")
        self.salmpling_frequency = 1
        self.max_planning_time = 3 # seconds

        # Parameters for the spatial model
        self.tau = 0.27 # The measurment noise
        self.phi_d = 530 # The spatial correlation length
        self.phi_t = 7200 # The temporal correlation length
        self.sigma = 2
        self.auv_speed = 1.6  # [m/s] The auv speed
        self.auv_depth = 0.5 # The depth layer the auv will be operating in
        self.reduce_points_factor = 2 # The factor to reduce the number of points added at each iteration
                                      # if factor is 1, no points will be removed
                                      # if factor is 2, half of the points will be removed
                                      # if factor is 3, 2/3 of the points will be removed                                      
        
        



        # s1: AUV setup
        self.auv = AUV()
        self.__loc_start = np.array(self.auv.get_vehicle_pos())
        print("[ACTION] Agent is set up")

        # s2: Operation field setup
        self.operation_field = Field()
        print("[ACTION] Operation field is set up ")

        # s3: Setting up prior field and auv data

        # Getting the experiment id
        if experiment_id == "new":
            # Finding a new experiment id
            # Read all the files in the folder 
            files = os.listdir("src/save_counter_data/")
            print("[INFO] expermimenets in memory")
            # finding the file numbers
            file_numbers = []
            for file in files:
                file_id = file.split("_")[-1]
                if file_id.isdigit():
                    file_numbers.append(int(file_id))
            #print(file_numbers)
            # Finding the lowest number that is not used
            for i in range(len(file_numbers) + 1):
                if i not in file_numbers:
                    experiment_id = i
                    break
            print("[INFO] Experiment id: ", experiment_id)
        print("[ACTION] Setting up prior")
        print("[INFO] Prior path: ", sinmod_file_name)
        self.prior = Prior(self.prior_path)
        print("[ACTION] Prior is set up")
        print("[INFO] First timestep in prior: ", datetime.datetime.fromtimestamp(self.prior.get_time_steps_seconds()[0]))
        print("[INFO] Last timestep in prior: ",datetime.datetime.fromtimestamp(self.prior.get_time_steps_seconds()[-1]))
        print("[ACTION] Setting up data handler")
        self.auv_data = AUVData(self.prior, 
                                phi_d=self.phi_d,
                                phi_t=self.phi_t,
                                tau=self.tau,
                                experiment_id=experiment_id,
                                time_diff_prior=self.diff_time_s)
        self.auv_data.load_most_recent_data() # This will load the most recent data if it exists
        self.descicion_rule = DescicionRule(self.descicion_rule_str)

        #



        # s4: storing variables
        self.__counter = 0
        self.time_planning = []
        self.time_start = time.time()
     


        print("[ACTION] Agent is set up")




    def run(self):

        
        # c1: start the operation from scratch.
        wp_depth = .5
        wp_start = self.wp_start

        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_submerged_time()
        popup_time = self.auv.get_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()

        # a1: move to current location
        lat, lon = WGS.xy2latlon(wp_start[1], wp_start[0])
        print("[ACTION] sending starting waypoint to auv")
        print("[INFO] lat: ",lat, " lon: ", lon)
        if self.start_from_current_location:
            
            current_position = self.auv.get_vehicle_pos()
            lat, lon = WGS.xy2latlon(current_position[0], current_position[1])
            self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)
        else:
            self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)
        print("[INFO] lat: ",lat, " lon: ", lon)
        print("[INFO] waypoint sent to auv")

        t_pop_last = time.time()
        update_time = rospy.get_time()


        # Setting up the data storage
        # This is the data we are getting from the vehicle
        position_data = []
        salinity_data = []
        time_data = []
        depth_data = []


        # Plann the first waypoint
        wp_next = np.empty(2)
        


        while not rospy.is_shutdown():
            if self.auv.init:

                t_now = time.time()

                print("counter: ", self.__counter, "\t vehicle state: ", self.auv.auv_handler.getState() ,end=" ")
                print(" \t time now: ", datetime.datetime.now(), end=" ")
                print(" \t prior time: ", datetime.datetime.fromtimestamp(time.time() - self.diff_time_s))


                # s1: append data
                loc_auv = self.auv.get_vehicle_pos() # Get the location of the vehicle
                position_data.append([loc_auv[1], loc_auv[0]])  # <--x- This is where we get the position data from the vehicle
                depth_data.append(loc_auv[2]) # <--- This is where we get the depth data from the vehicle
                salinity_data.append(self.auv.get_salinity()) # <--- This is where we get the salinity data from the vehicle
                time_data.append(time.time())  # <--- This is where we get the time data from the vehicle
        


                # Check if the vehicle is waiting for a new waypoint
                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        print("[ACTION] Popping up]")
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        print("[ACTION] pop-up message sent")
                        
                        # TODO: do correction here
                        t_pop_last = time.time()

                    # Timming the planning
                    t_plan_start = time.time()

                   

                    # Checking if the points are legal
                    illigal_points = []
                    for s in position_data:
                        if not self.operation_field.is_loc_legal(np.flip(np.array(s))):
                            illigal_points.append(s)
                    if len(illigal_points) > 0:
                        print("[WARNING] illigal points found")
                        print("[INFO] there are" , len(illigal_points), "illigal points out of", len(position_data))

                     # update the points in memory
                    self.auv_data.add_new_datapoints(np.array(position_data), np.array(time_data), np.array(salinity_data))
                    
                    # Reset the data storage
                    position_data = []
                    salinity_data = []
                    time_data = []
                    depth_data = []

                    # Get the next waypoint
                    if self.__counter == 0:
                        wp_next = self.plan_first_waypoint()
                    else:
                        wp_next = self.plan_next_waypoint(wp_next)


                    # Going from x,y to lat,lon in the WGS84 system
                    lat, lon = WGS.xy2latlon(wp_next[1], wp_next[0])


                    # Set the waypoint to the vehicle 
                    print("[ACTION] Setting waypoint]")
                    print(f"[INFO] lat: {lat}, lon: {lon}")
                    self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)
                    print("[ACTION] Waypoint sent to auv_handler")

                    # Update the time planning
                    self.time_planning.append(time.time() - t_plan_start)

                    if self.time_planning[-1] > self.max_planning_time:
                        print("[INFO] Planning took too long, will down sample the points")
                        print("[INFO] Points before: ", self.auv_data.get_number_of_points_in_memory())
                        self.auv_data.down_sample_points()
                        print("[INFO] Points after: ", self.auv_data.get_number_of_points_in_memory())

                    # Update the counter 
                    print("-----------------------------------------------------")
                    print("#################   Counter", self.__counter + 1, "   #################")
                    print("-----------------------------------------------------")
                    self.__counter += 1
                
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()




    def plan_next_waypoint(self , a) -> np.ndarray:

        # a is the current waypoint in x, y
        # b is the next waypoint in x, y


        time_start = time.time()

        directions = np.linspace(0,2 * np.pi, self.n_directions + 1) + np.random.rand(1) * np.pi 
        end_points = []
        for theta in directions:
            b = np.array([a[0] + self.horizion * np.cos(theta), a[1] + self.horizion * np.sin(theta)]).ravel()   

            a_prev = self.auv_data.auv_data["path_list"][-2]


            u = b - a
            v = a_prev - a
            c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
            angle = np.arccos(np.clip(c, -1, 1)) # if you really want the angle

            # Nu U-turn, the higher the number the higher the movement freedom 
            if angle  > np.pi/6:

                if self.operation_field.is_path_legal(np.flip(a),np.flip(b)):
                    end_points.append(b)
                    #plt.plot([a[0], b[0]],[a[1], b[1]], c="green")
                else:
                    closest_legal_point = np.flip(self.operation_field.get_closest_intersect_point(np.flip(a),np.flip(b)))
                
                    dist_ab = np.linalg.norm(closest_legal_point - a)

                    if dist_ab > self.radius:
                        end_points.append(closest_legal_point)

        if len(end_points) == 0:
            # Noe legal points found

            print("[WARNING] No legal points found, will try again ")
            directions = np.linspace(0,2 * np.pi, self.n_directions + 1) + np.random.rand(1) * np.pi 
            end_points = []
            for theta in directions:
                b = np.array([a[0] + self.horizion * np.cos(theta), a[1] + self.horizion * np.sin(theta)]).ravel()   

                a_prev = self.auv_data.auv_data["path_list"][-2]
                u = b - a
                v = a_prev - a
                c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
                angle = np.arccos(np.clip(c, -1, 1)) # if you really want the angle

                # Nu U-turn, the higher the number the higher the movement freedom 

                if self.operation_field.is_path_legal(np.flip(a),np.flip(b)):
                    end_points.append(b)
                    #plt.plot([a[0], b[0]],[a[1], b[1]], c="green")
                else:
                    closest_legal_point = np.flip(self.operation_field.get_closest_intersect_point(np.flip(a),np.flip(b)))
                
                    dist_ab = np.linalg.norm(closest_legal_point - a)

                    if dist_ab > self.radius:
                        end_points.append(closest_legal_point)
            


        # Getting the direction data
        direction_data = self.auv_data.predict_directions(end_points) 

        # Finding which direction is the best
        descicion = self.descicion_rule.descicion_rule(direction_data, self.auv_data.auv_data)

        # Store the direction data
        with open("src/save_counter_data/" + str(self.experiment_id) + "/" + "direction_data_" + str(self.__counter), 'wb') as f:
            pickle.dump(direction_data, f)
        # Store the descicion data
        with open("src/save_counter_data/" + str(self.experiment_id) + "/" + "descicion_data_" + str(self.__counter), 'wb') as f:
            pickle.dump(descicion, f)

        b = descicion["end_point"] 
        dist_ab = np.linalg.norm(b - a)
        b = a + (b - a) / dist_ab  * min(dist_ab, self.radius)

        descicion["end_point"] = b

        curr_time = self.auv_data.auv_data["T"][-1]

        time_end = time.time()

        self.time_planning.append(time_end - time_start)
        print("[TIMING] \t Planning took: ", time_end - time_start)
        return b
    
    def plan_first_waypoint(self) -> np.ndarray:
        loc_start = self.__loc_start
        a = np.array([loc_start[1], loc_start[0]]).ravel()
        theta =  np.random.rand(1) * np.pi * 2 
        b = np.array([a[0] + self.radius_init * np.cos(theta), a[1] + self.radius_init * np.sin(theta)]).ravel()
        print("a", a)
        print("b", b)
        while self.operation_field.is_path_legal(np.flip(a),np.flip(b)) == False: 
            theta =  np.random.rand(1) * np.pi * 2 
            b = np.array([a[0] + self.radius_init * np.cos(theta), a[1] + self.radius_init * np.sin(theta)]).ravel()
        return b
    
    def save_parameters(self):
        
        parameter_dict = {}
        parameter_dict["radius"] = self.radius
        parameter_dict["radius_init"] = self.radius_init
        parameter_dict["horizion"] = self.horizion
        parameter_dict["n_directions"] = self.n_directions
        parameter_dict["descicion_rule"] = self.descicion_rule_str
        parameter_dict["prior_path"] = self.prior_path
        parameter_dict["date_today"] = self.date_today
        parameter_dict["prior_date"] = self.prior_date
        parameter_dict["high_tide_prior"] = self.high_tide_prior
        parameter_dict["high_tide_today"] = self.high_tide_today
        parameter_dict["tau"] = self.tau
        parameter_dict["pid_d"] = self.pid_d
        parameter_dict["phi_t"] = self.phi_t
        parameter_dict["sigma"] = self.sigma
        parameter_dict["auv_speed"] = self.auv_speed
        parameter_dict["auv_depth"] = self.auv_depth
        parameter_dict["reduce_points_factor"] = self.reduce_points_factor
        parameter_dict["max_planning_time"] = self.max_planning_time

        # The path we want to save the data to       
        path = "src/save_counter_data/" + str(self.experiment_id) + "/"

        # Checking if the folder exists
        if not os.path.exists(path):
            os.makedirs(path)

        print("[ACTION] Saving the parmeters")

        # Saving the data
        with open(path + "parameters", 'wb') as f:
            pickle.dump(parameter_dict, f)





        
    
if __name__ == "__main__":

    experiment_id = "new"
    Agent = Agent(experiment_id=experiment_id)
    Agent.run()








                    

