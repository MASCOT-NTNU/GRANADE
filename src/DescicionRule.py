import numpy as np
import random
from scipy.stats import norm


class DescicionRule:

    def __init__(self, rule: str) -> None:
        self.rule = rule
        

    def descicion_rule(self, direction_data, auv_data):
        
        if self.rule == "max_prob":
            return self.descicion_rule_max_prob(direction_data, auv_data)
        
        if self.rule == "random":
            return self.descicion_rule_random(direction_data, auv_data)
        
        if self.rule == "look_far":
            return self.descicion_rule_look_far(direction_data, auv_data)
        
        if self.rule == "avg_prob":
            return self.descicion_rule_avg_prob(direction_data, auv_data)
        
        if self.rule == "max_improvement":
            return self.descicion_max_expected_improvement(direction_data, auv_data)
        
        if self.rule == "top_p_improvement":
            return self.descicion_top_p_improvement(direction_data, auv_data)
        
        if self.rule == "avg_improvement":
            return self.descicion_avg_expected_improvement(direction_data, auv_data)
        
        if self.rule == "max_reduction":
            return self.descicion_max_expected_reduction(direction_data, auv_data)
        
        if self.rule == "top_p_probability":
            return self.descicion_rule_top_p_prob(direction_data, auv_data)
        
        if self.rule == "avg_variance":
            return self.highest_avg_variance(direction_data, auv_data)
        
        if self.rule == "max_variance":
            return self.highest_max_variance(direction_data, auv_data)
        
        print(self.rule," is not a method")
        

        
    def descicion_rule_max_prob(self, direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["S"][-1]
        
        best_direction_ind = 0
        prob_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            prob_new_max = 1 - norm.cdf((max_g - G)/np.sqrt(Var_G)) + norm.cdf((-max_g - G)/np.sqrt(Var_G))
            max_prob_gradient = np.max(prob_new_max)
            max_ind = np.argmax(prob_new_max)



            if np.max(prob_new_max) > prob_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                prob_max = np.max(prob_new_max)
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_rule_avg_prob(self, direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        avg_prob_best = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            prob_new_max = 1 - norm.cdf((max_g - G)/np.sqrt(Var_G)) + norm.cdf((-max_g - G)/np.sqrt(Var_G))
            avg_prob_dir = np.average(prob_new_max)



            if avg_prob_dir > avg_prob_best and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                avg_prob_best = avg_prob_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_rule_top_p_prob(self, direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 

        
        a = auv_data["path_list"][-1]
        
        p = 0.2
        
        best_direction_ind = 0
        avg_prob_best = 0
        best_end_point = direction_data["end_points"][0]
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            prob_new_max = 1 - norm.cdf((max_g - G)/np.sqrt(Var_G)) + norm.cdf((-max_g - G)/np.sqrt(Var_G))
            
            k = int(p * len(prob_new_max))
            avg_prob_dir = np.average(np.sort(prob_new_max)[-k:])



            if avg_prob_dir > avg_prob_best and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                avg_prob_best = avg_prob_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion

    def descicion_max_expected_improvement(self,direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 

        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        improvement_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
        
            Sd_G = np.sqrt(Var_G)

            #div_A = np.max(np.array([norm.cdf((- max_g - G) / Sd_G), np.ones(len(G)) * 10**(-32)]), 0)
            #div_B = np.max(np.array([(1 - norm.cdf((max_g - G) / Sd_G)), np.ones(len(G)) * 10**(-32)]), 0)

            #A = norm.pdf((-max_g - G) / Sd_G) / div_A
            #B = norm.pdf((max_g - G) / Sd_G) / div_B
            #expected_improvement = Sd_G * (A + B)
            
            I_pluss = (G - max_g) * (1 - norm.cdf((max_g - G) / Sd_G))  + Sd_G * norm.pdf((max_g -G) / Sd_G)
            I_minus = (-G - max_g) * norm.cdf((- max_g - G) / Sd_G)  + Sd_G * norm.pdf((-max_g - G) / Sd_G)

            expected_improvement = I_pluss + I_minus
            
            best_improvement_dir = np.max(expected_improvement)



            if best_improvement_dir > improvement_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                improvement_max = best_improvement_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_avg_expected_improvement(self,direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        average_improvement_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
        
            Sd_G = np.sqrt(Var_G)
            
            I_pluss = (G - max_g) * (1 - norm.cdf((max_g - G) / Sd_G))  + Sd_G * norm.pdf((max_g -G) / Sd_G)
            I_minus = (-G - max_g) * norm.cdf((- max_g - G) / Sd_G)  + Sd_G * norm.pdf((-max_g - G) / Sd_G)

            expected_improvement = I_pluss + I_minus
            
            avg_improvement_dir = np.average(expected_improvement)



            if avg_improvement_dir > average_improvement_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                average_improvement_max = avg_improvement_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_top_p_improvement(self, direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["S"][-1]
        
        p = 0.2
        
        best_direction_ind = 0
        average_improvement_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
        
            Sd_G = np.sqrt(Var_G)
            
            I_pluss = (G - max_g) * (1 - norm.cdf((max_g - G) / Sd_G))  + Sd_G * norm.pdf((max_g -G) / Sd_G)
            I_minus = (-G - max_g) * norm.cdf((- max_g - G) / Sd_G)  + Sd_G * norm.pdf((-max_g - G) / Sd_G)

            expected_improvement = I_pluss + I_minus
            
            k = int(p * len(expected_improvement))
            
            avg_improvement_dir = np.average(np.sort(expected_improvement)[-k:])


            if avg_improvement_dir > average_improvement_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                average_improvement_max = avg_improvement_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion



    def highest_avg_variance(self,direction_data, auv_data):
        n = direction_data["n_directions"]

        
        a = auv_data["S"][-1]
        
        best_direction_ind = 0
        average_variance_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            avg_variance_dir = np.average(Var_G)



            if average_variance_max > avg_variance_dir and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                average_variance_max = avg_variance_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def highest_max_variance(self,direction_data, auv_data):
        n = direction_data["n_directions"]

        
        a = auv_data["S"][-1]
        
        best_direction_ind = 0
        max_variance = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            max_variance_dir = np.max(Var_G)



            if max_variance > max_variance_dir and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                max_variance = max_variance_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_max_expected_reduction(self,direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        reduction_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            
            Sd_G = np.sqrt(Var_G)
            abs_G = np.abs(G)

            A1 = (abs_G - G) * (norm.cdf( (abs_G - G) / Sd_G ) - norm.cdf( - (G) / Sd_G))
            A2 = Sd_G * (norm.pdf((abs_G-G)/Sd_G) -norm.pdf((-G)/Sd_G))
            B1 = (abs_G + G) * (norm.cdf(-G/Sd_G) - norm.cdf( (-abs_G-G) / Sd_G))
            B2 = Sd_G * (norm.pdf(- G / Sd_G) -norm.pdf((-abs_G - G)/Sd_G))
            expected_reduction = A1 + B1
            
            max_reduction_dir = np.average(expected_reduction)



            if max_reduction_dir > reduction_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                reduction_max = max_reduction_dir
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion
        
        
    def descicion_max_prob_reduction(self,direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        prob_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            prob_new_max = 1 - norm.cdf((max_g - G)/np.sqrt(Var_G)) + norm.cdf((-max_g - G)/np.sqrt(Var_G))
            max_prob_gradient = np.max(prob_new_max)
            max_ind = np.argmax(prob_new_max)



            if np.max(prob_new_max) > prob_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                prob_max = np.max(prob_new_max)
                best_end_point = b

            ###################################
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion


    def descicion_rule_look_far(self,direction_data, auv_data):
        n = direction_data["n_directions"]
        G = auv_data["G"]
        max_g = np.max(np.abs(G)) 
        
        a = auv_data["path_list"][-1]
        
        best_direction_ind = 0
        prob_max = 0
        best_end_point = direction_data["end_points"][0]
        
        for j in range(n):
            G = direction_data["gradient_directions"][j]
            Var_G = direction_data["var_gradient_directions"][j]
            b = direction_data["end_points"][j]
            
            prob_new_max = 1 - norm.cdf((max_g - G)/np.sqrt(Var_G)) + norm.cdf((-max_g - G)/np.sqrt(Var_G))
            max_prob_gradient = np.max(prob_new_max)
            max_ind = np.argmax(prob_new_max)



            if np.max(prob_new_max) > prob_max and np.linalg.norm(b-a) > 5:
                best_direction_ind = j
                prob_max = np.max(prob_new_max)
                best_end_point = b

            ###################################
        
        best_end_point = a + (best_end_point - a) / np.linalg.norm(best_end_point - a) * 10
        
        descicion = {"end_point": best_end_point, "direction_ind": best_direction_ind}
        return descicion
            

    def descicion_rule_random(self,direction_data, auv_data):
        
        # The current location
        a = auv_data["path_list"][-1]
        
        possible_descisions = []
        for i, b in enumerate(direction_data["end_points"]):
            if np.linalg.norm(b - a) > 5:
                possible_descisions.append({"end_point": b, "direction_ind": i})
        
        
        
        descicion = random.sample(possible_descisions, k=1)[0]
        return descicion

        
        