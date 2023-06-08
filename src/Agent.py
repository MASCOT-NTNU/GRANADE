
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

    def __init__(self) -> None:
        """
        Setting up the agent 
        """

        # Thses are the parameters for the mission
        self.n_directions = 8
        self.horizion = 1000
        self.radius = 250
        self.descicion_rule = "top_p_improvement"


        # s1: AUV setup
        self.auv = AUV()
        self.__loc_start = np.array(self.auv.get_vehicle_pos())
        print("Agent is set up")

        # s2: Operation field setup
        self.operation_field = Field()
        print(" Operation field is set up ")

        # s3: Setting up prior field and auv data
        self.prior = Prior()
        self.auv_data = AUVData(self.prior)
        self.desicion_rule = DescicionRule(self.descicion_rule)



        speed = self.auv.get_speed()


        # s4: storing variables
        self.__counter = 0
        self.max_planning_time = 3 # seconds
        self.time_planning = []

    
        pass


    def run(self):

        
        # c1: start the operation from scratch.
        wp_depth = .5
        wp_start = self.planner.get_starting_waypoint()
        wp_end = self.planner.get_end_waypoint()     # This is not something 

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

        ctd_data = []

        while not rospy.is_shutdown():
            if self.auv.init:

                t_now = time.time()

                print("counter: ", self.__counter)

                # s1: append data
                loc_auv = self.auv.get_vehicle_pos()
                ctd_data.append([loc_auv[0], loc_auv[1], loc_auv[2], self.auv.get_salinity()])  # <--- This is where we get the data



                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                    

                    # Get the next waypoint
                    wp_next = self.plan_next_waypoint()

                    if self.time_planning[-1] > self.max_planning_time:
                        print("Planning took too long, will down sample the points")
                        self.auv_data.down_sample_points()

                    # Set the waypoint to the vehicle 
                    self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)



    def plan_next_waypoint(self) -> np.ndarray:

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
        print("Planning took: ", time_end - time_start)

        return b







                    

