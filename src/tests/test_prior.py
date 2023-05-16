"""
Test prior module
"""
import os
from unittest import TestCase
from Prior import Prior
import numpy as np
import matplotlib.pyplot as plt
from numpy import testing


class TestPrior(TestCase):

    def setUp(self) -> None:
        # print(os.listdir(os.getcwd() + "/../sinmod/"))
        filepath = os.getcwd() + "/../sinmod/samples_2022.05.04.nc"
        self.prior = Prior(filepath)

    # def test_load_sinmod_data(self) -> None:
    #
    #     # c1, test
    #
    #     pass

    def test_get_salinity_values(self) -> None:

        sampling_frequency = 0.1 # s^-1
        auv_speed = 1.6 # m/2
        t_0 = self.prior.get_time_steps_seconds()[70] # start time 
        a = np.array([0,2000]) # start point
        b = np.array([1000,3000]) # end point

        def get_points(a,b,t_0):
        
            dist = np.linalg.norm(b - a)
            total_time = dist / auv_speed
            n_points = int(total_time * sampling_frequency)
            t_end = t_0 + total_time
            
            T = np.linspace(t_0, t_end, n_points)
            S = np.linspace(a, b, n_points)
            
            return S, T

        S, T = get_points(a,b,t_0)

        salinity = self.prior.get_salinity_S_T(S, T)

        if np.count_nonzero(np.isnan(salinity)) > 0:
            raise print("nan values")

        self.assertEqual(T.shape, salinity.shape)

    # def test_show_result(self) -> None:
    #     # s1, load sinmod values
    #
    #     # s2, visualize sinmod values


        
                            



    