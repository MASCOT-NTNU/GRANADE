"""
Field module for the GRANADE project.

Objective:
    - Create boundary for the field.
    - Create obstacles for the field.
    - Evaluate the field at a given location.
"""
from WGS import WGS
from shapely.geometry import Point, LineString, Polygon
import numpy as np
from math import cos, sin, radians
import pandas as pd
import os
import matplotlib.pyplot as plt


class Field:

    def __init__(self) -> None:
        # s1, load csv files for all the polygons
        self.__wgs_polygon_border = pd.read_csv(os.getcwd() + "/src/csv/polygon_border.csv").to_numpy()
        self.__wgs_polygon_obstacle = pd.read_csv(os.getcwd() + "/src/csv/polygon_obstacle.csv").to_numpy()

        # s2, convert wgs to xy
        lat, lon = WGS.latlon2xy(self.__wgs_polygon_border[:, 0], self.__wgs_polygon_border[:, 1])
        self.__polygon_border = np.stack((lat, lon), axis=1)
        lat, lon = WGS.latlon2xy(self.__wgs_polygon_obstacle[:, 0], self.__wgs_polygon_obstacle[:, 1])
        self.__polygon_obstacle = np.stack((lat, lon), axis=1)

        # s3, create shapely polygon objects
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__polygon_obstacle_shapely = Polygon(self.__polygon_obstacle)

        # s4, create shapely line objects
        self.__line_border_shapely = LineString(self.__polygon_border)
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)

        # s5, create river outlets
        river_outlets_lat_lon = np.array([[63.436711, 10.391483] , [63.446320, 10.417712], [63.433722, 10.376099], [63.433617, 10.365583]])
        river_outlets_xy = np.array([WGS.latlon2xy(river_outlets_lat_lon[:, 0], river_outlets_lat_lon[:, 1])]).T

    def is_loc_legal(self, loc: np.ndarray) -> bool:
        """
        Check if a point is legal:
        That means that the point is inside the operational area and not inside the obstacle
        """
        if self.__border_contains(loc):
            if self.__obstacle_contains(loc):
                return False
            else:
                if self.__is_loc_on_edge_of_obstacle(loc):
                    return False
                else:
                    return True
        else:
            return False

    def __border_contains(self, loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)

    def __obstacle_contains(self, loc: np.ndarray) -> bool:
        """ Test if obstacle contains the point. """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_obstacle_shapely.contains(point)

    def __is_loc_on_edge_of_obstacle(self, loc: np.ndarray) -> bool:
        """ Test if loc is on the edge of the obstacle. """
        x, y = loc
        point = Point(x, y)
        return point.touches(self.__polygon_obstacle_shapely)

    def is_path_legal(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if a path is legal:
        That means that the path does not intersect with the border or the obstacle
        """
        if self.__is_border_in_the_way(loc_start, loc_end) or self.__is_obstacle_in_the_way(loc_start, loc_end):
            return False
        else:
            if self.is_loc_legal(loc_start) and self.is_loc_legal(loc_end):
                return True
            else:
                return False
            
    def __is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_border_shapely.intersects(line)
    
    def __is_obstacle_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_obstacle_shapely.intersects(line)

    def get_closest_intersect_point(self, loc_start: np.ndarray, loc_end: np.ndarray) -> np.ndarray:

        # These are the intersect points
        intersect_points = self.__get_path_intersect(loc_start, loc_end)

        closest_point = np.empty(2)
        if len(intersect_points) > 0:
            dist_list = np.linalg.norm(loc_start - intersect_points, axis=1)

            idx = dist_list.argmin()
            closest_point = intersect_points[idx]

        return closest_point

    def __get_path_intersect(self,loc_start: np.ndarray, loc_end: np.ndarray) -> np.ndarray:
        """
        Returns all the intersection points with a path with the border or obstacle         
        """

        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])

        # The line intersects will eighter return 0, 1 or multilple points
        intersect_points_border = self.__line_border_shapely.intersection(line)
        intersect_points_obstacle = self.__line_obstacle_shapely.intersection(line)

        intersect_points_border_np = np.array(intersect_points_border)
        intersect_points_obstacle_np = np.array(intersect_points_obstacle)

        m = len(intersect_points_border_np.reshape(-1,1))/2 + len(intersect_points_obstacle_np.reshape(-1,1))/2
        intersect_points = np.empty((int(m),2))
        
        k = 0
        if len(intersect_points_border_np) > 0:
            if len(intersect_points_border_np.reshape(-1,1)) == 2:
                # This is a single point
                intersect_points[k] = intersect_points_border_np
                k += 1
            else:
                for p in intersect_points_border_np:
                    intersect_points[k] = p
                    k += 1

         
        if len(intersect_points_obstacle_np) > 0:
            if len(intersect_points_obstacle_np.reshape(-1,1)) == 2:
                # This is a single point
                intersect_points[k] = intersect_points_obstacle_np
                k += 1
            else:
                for p in intersect_points_obstacle_np:
                    intersect_points[k] = p
                    k += 1

        return np.array(intersect_points)
    
    def get_polygon_border(self) -> np.ndarray:
        return self.__polygon_border

    def get_polygon_obstacle(self) -> np.ndarray:
        return self.__polygon_obstacle
    
    def get_exterior_border(self) -> np.ndarray:
        return self.__polygon_border_shapely.exterior.xy
    
    def get_exterior_obstacle(self) -> np.ndarray:
        return self.__polygon_obstacle_shapely.exterior.xy
    
    # REMOVE
    def plot_operational_area(self, show=True) -> None:
        """
        This functions plots the border and the operational area. 
        """

        polygon_obstacle =  self.__polygon_obstacle_shapely
        polygon_border = self.__polygon_border_shapely
        xo, yo = polygon_obstacle.exterior.xy
        xb, yb = polygon_border.exterior.xy
        plt.plot(yo, xo, label="obstacle", c="orange")
        plt.plot(yb, xb, label="border", c="green")
        #plt.legend()
        if show:
            plt.show()

    # REMOVE
    def add_plot_operational_area(self,axs) -> None:
        """
        This functions plots the border and the operational area. 
        """

        polygon_obstacle =  self.__polygon_obstacle_shapely
        polygon_border = self.__polygon_border_shapely
        xo, yo = polygon_obstacle.exterior.xy
        xb, yb = polygon_border.exterior.xy
        #self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        axs.plot(yo, xo, label="obstacle", c="orange")
        axs.plot(yb, xb, label="border", c="green")


if __name__ == "__main__":
    f = Field()