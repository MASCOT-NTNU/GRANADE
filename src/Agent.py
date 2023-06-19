
import numpy as np
import os
import time 
import math 
import rospy

from AUV import AUV
from WGS import WGS
from Field import Field
from Prior import Prior
from AUV_data import AUVData
from DescicionRule import DescicionRule





class Agent:

    def __init__(self,
                 experiment_id = "new") -> None:
        """
        Setting up the agent 
        """
        print("[ACTION] Setting up the agent")

        # Thses are the parameters for the mission
        self.n_directions = 8
        self.horizion = 1000  # meters, how far the agent can see
        self.radius = 250 # meters, how far the agent will move 
        self.radius_init = 150 # meters, how far the agent will move on the first iteration
        self.descicion_rule = "top_p_improvement"
        self.prior_path = "/src/sinmod_files/" + 'samples_2022.05.04.nc'
        
        self.salmpling_frequency = 1

        # Parameters for the spatial model
        self.tau = 0.27 # The measurment noise
        self.phi_d = 530 # The spatial correlation length
        self.phi_t = 7200 # The temporal correlation length
        self.sigma = 2
        self.auv_speed = 1.5  # [m/s] The auv speed
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
        self.prior = Prior(self.prior_path)
        print("[ACTION] Prior is set up")
        print("[ACTION] Setting up data handler")
        self.auv_data = AUVData(self.prior, 
                                phi_d=self.phi_s,
                                phi_t=self.phi_t,
                                tau=self.tau,
                                experiment_id=experiment_id)
        self.auv_data.load_most_recent_data() # This will load the most recent data if it exists
        self.desicion_rule = DescicionRule(self.descicion_rule)



        # s4: storing variables
        self.__counter = 0
        self.max_planning_time = 3 # seconds
        self.time_planning = []
        self.time_start = time.time()

        print("[ACTION] Agent is set up")




    def run(self):

        
        # c1: start the operation from scratch.
        wp_depth = .5
        wp_start = self.planner.get_starting_waypoint()

        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_max_submerged_time()
        popup_time = self.auv.get_min_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()

        # a1: move to current location
        lat, lon = WGS.xy2latlon(wp_start[0], wp_start[1])
        self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)

        t_pop_last = time.time()
        update_time = rospy.get_time()


        # Setting up the data storage
        # This is the data we are getting from the vehicle
        position_data = []
        salinity_data = []
        time_data = []
        depth_data = []


        # Plann the first waypoint
        


        while not rospy.is_shutdown():
            if self.auv.init:

                t_now = time.time()

                print("counter: ", self.__counter)

                # s1: append data
                loc_auv = self.auv.get_vehicle_pos() # Get the location of the vehicle
                position_data.append([loc_auv[1], loc_auv[0]])  # <--- This is where we get the position data from the vehicle
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

                    # update the points in memory
                    self.auv_data.add_new_datapoints(np.array(position_data), np.array(time_data), np.array(salinity_data))
                    
                    # Reset the data storage
                    position_data = []
                    salinity_data = []
                    time_data = []
                    depth_data = []

                    # Get the next waypoint
                    wp_next = np.empty(2)
                    if self.__counter == 0:
                        wp_next = self.plan_first_waypoint()
                    else:
                        wp_next = self.plan_next_waypoint()


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
                    print("#################   Counter", self.__counter, "   #################")
                    print("-----------------------------------------------------")
                    self.__counter += 1
                
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()




    def plan_next_waypoint(self , a) -> np.ndarray:


        time_start = time.time()

        directions = np.linspace(0,2 * np.pi, self.n_directions + 1) + np.random.rand(1) * np.pi 
        end_points = []
        for theta in directions:
            b = np.array([a[0] + self.horizion * np.cos(theta), a[1] + self.horizion * np.sin(theta)]).ravel()   

            a_prev = self.AUV_data.auv_data["path_list"][-2]


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

                a_prev = self.AUV_data.auv_data["path_list"][-2]


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
        direction_data = self.AUV_data.predict_directions(end_points) 

        # Finding which direction is the best
        descicion = self.descicion_rule.descicion_rule(direction_data, self.AUV_data.auv_data)

        b = descicion["end_point"] 
        dist_ab = np.linalg.norm(b - a)
        b = a + (b - a) / dist_ab  * min(dist_ab, self.radius)

        descicion["end_point"] = b

        curr_time = self.AUV_data.auv_data["T"][-1]

        time_end = time.time()

        self.time_planning.append(time_end - time_start)
        print("[TIMING] \t Planning took: ", time_end - time_start)

        return b
    
    def plan_first_waypoint(self) -> np.ndarray:
        a = self.__loc_start
        b = np.array([a[1] + self.radius_init * np.cos(theta), a[0] + self.radius_init * np.sin(theta)]).ravel()
        while self.operation_field.is_path_legal(np.flip(a),np.flip(b)) == False: 
            theta =  np.random.rand(1) * np.pi * 2 
            b = np.array([a[0] + self.radius_init * np.cos(theta), a[1] + self.radius_init * np.sin(theta)]).ravel()
        return b
    
if __name__ == "__main__":

    Agent = Agent()
    Agent.run()








                    

