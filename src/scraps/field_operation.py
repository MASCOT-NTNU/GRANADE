
"""
Field handles field discretization and validation.
- generate grid discretization.
- check legal conditions of given locations.
- check collision with obstacles.
"""
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from typing import Union

from Config import Config


class FieldOperation:
    """
    Field handles everything with regarding to the field element.
    """
    def __init__(self, neighbour_distance: float = 120):
        # config element
        self.__config = Config()
        self.__neighbour_distance = neighbour_distance  # metres between neighbouring locations. # REMOVE

        # border element
        self.__polygon_border = self.__config.get_polygon_border()
        self.__polygon_border_shapely = self.__config.get_polygon_border_shapely()
        self.__line_border_shapely = LineString(self.__polygon_border)

        # obstacle element
        self.__polygon_obstacle = self.__config.get_polygon_obstacle()
        self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)
        
        """ Get the xy limits and gaps for the bigger box """
        xb = self.__polygon_border[:, 0]
        yb = self.__polygon_border[:, 1]
        self.__xmin, self.__ymin = map(np.amin, [xb, yb])
        self.__xmax, self.__ymax = map(np.amax, [xb, yb])
        self.__xlim = np.array([self.__xmin, self.__xmax])
        self.__ylim = np.array([self.__ymin, self.__ymax])
        self.__ygap = self.__neighbour_distance * cos(radians(60)) * 2
        self.__xgap = self.__neighbour_distance * sin(radians(60))

        # grid element
        self.__grid = np.empty([0, 2]) # REMOVE
        self.__construct_grid() # REMOVE

        # neighbour element
        self.__neighbour_hash_table = dict() # REMOVE
        self.__construct_hash_neighbours() # REMOVE

    # REMOVE
    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def border_contains(self, loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)

    def obstacle_contains(self, loc: np.ndarray) -> bool:
        """ Test if obstacle contains the point. """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_obstacle_shapely.contains(point)


    def is_loc_legal(self, loc: np.ndarray) -> bool:
        """
        Check if a point is legal: 
        That means that the point is inside the operational area and not inside the obstacle 
        """
        if self.border_contains(loc):
            if self.obstacle_contains(loc):
                return False
            else:
                return True
        else:
            return False

    def is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_border_shapely.intersects(line)

    def is_obstacle_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_obstacle_shapely.intersects(line)

    
    def is_path_legal(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if a path is legal: 
        That means that the path does not intersect with the border or the obstacle 
        """
        if self.is_border_in_the_way(loc_start, loc_end) or self.is_obstacle_in_the_way(loc_start, loc_end):
            return False
        else:
            if self.border_contains(loc_start) and self.border_contains(loc_end):
                return True
            else:
                return False
       
    def get_path_intersect(self,loc_start: np.ndarray, loc_end: np.ndarray):
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

        return intersect_points

    def get_closest_intersect_points(self,loc_start: np.ndarray, loc_end: np.ndarray) -> np.ndarray:

        # These are the intersect points 
        intersect_points = self.get_path_intersect(loc_start, loc_end)
    
        closest_point = np.empty(2)
        if len(intersect_points) > 0:
    
            dist_list = np.linalg.norm(loc_start - intersect_points, axis=1)

            idx = dist_list.argmin()
            closest_point = intersect_points[idx]

        return  closest_point





    # REMOVE
    def __construct_grid(self) -> None:
        """ Construct the field grid based on the instruction given above.
        - Construct regular meshgrid.
        .  .  .  .
        .  .  .  .
        .  .  .  .
        - Then move the even row to the right side.
        .  .  .  .
          .  .  .  .
        .  .  .  .
        - Then remove illegal locations.
        - Then add the depth layers.
        """
        gx = np.arange(self.__xmin, self.__xmax, self.__xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(self.__ymin, self.__ymax, self.__ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gx)):
            for j in range(len(gy)):
                if i % 2 == 0:
                    y = gy[j] + self.__ygap / 2
                    x = gx[i]
                else:
                    y = gy[j]
                    x = gx[i]
                loc = np.array([x, y])
                if self.border_contains(loc) and not self.obstacle_contains(loc):
                    grid2d.append([x, y])
                    counter_grid2d += 1
        self.__grid = np.array(grid2d)
    
    # REMOVE
    def __construct_hash_neighbours(self) -> None:
        """ Construct the hash table for containing neighbour indices around each waypoint.
        - Directly use the neighbouring radius to determine the neighbouring indices.
        """
        no_grid = self.__grid.shape[0]
        ERROR_BUFFER = .01 * self.__neighbour_distance
        for i in range(no_grid):
            xy_c = self.__grid[i].reshape(1, -1)
            dist = cdist(self.__grid, xy_c)
            ind_n = np.where((dist <= self.__neighbour_distance + ERROR_BUFFER) *
                             (dist >= self.__neighbour_distance - ERROR_BUFFER))[0]
            self.__neighbour_hash_table[i] = ind_n

    # REMOVE
    def get_neighbour_indices(self, ind_now: Union[int, np.ndarray]) -> np.ndarray:
        """ Return neighbouring indices according to given current index. """
        if type(ind_now) == np.int64:
            return self.__neighbour_hash_table[ind_now]
        else:
            neighbours = np.empty([0])
            for i in range(len(ind_now)):
                idnn = self.__neighbour_hash_table[ind_now[i]]
                neighbours = np.append(neighbours, idnn)
            return np.unique(neighbours.astype(int))

    # REMOVE
    def get_grid(self) -> np.ndarray:
        """
        Returns: grf grid.
        """
        return self.__grid

    # REMOVE
    def get_neighbour_distance(self) -> float:
        """ Return neighbour distance. """
        return self.__neighbour_distance

    # REMOVE
    def get_location_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return self.__grid[ind, :]

    # REMOVE
    def get_ind_from_location(self, location: np.ndarray) -> Union[np.ndarray, None]:
        """
        Args:
            location: np.array([xp, yp])
        Returns: index of the closest waypoint.
        """

        if len(location) > 0:
            dm = location.ndim
            if dm == 1:
                d = cdist(self.__grid, location.reshape(1, -1))
                return np.argmin(d, axis=0)
            elif dm == 2:
                d = cdist(self.__grid, location)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_border_limits(self):
        return self.__xlim, self.__ylim

    
    def plot_operational_area(self, show=True) -> None:
        """
        This functions plots the border and the operational area. 
        """

        polygon_obstacle =  self.__polygon_obstacle_shapely
        polygon_border = self.__polygon_border_shapely
        xo, yo = polygon_obstacle.exterior.xy
        xb, yb = polygon_border.exterior.xy
        self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        plt.plot(yo, xo, label="obstacle", c="orange")
        plt.plot(yb, xb, label="border", c="green")
        #plt.legend()
        if show:
            plt.show()

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
    f = FieldOperation()

    n = 200
    random_x = np.random.uniform(-2500, 2000, size = n)
    random_y = np.random.uniform(1000, 5000, size = n)
    random_points = np.array([random_y,random_x]).T
    for s in random_points:
        if f.is_loc_legal(s):
            plt.scatter(s[1],s[0], c="green")
        else:
            plt.scatter(s[1],s[0], c="red")
        

    f.plot_operational_area()
    
    n = 20
    random_x = np.random.uniform(-2500, 2000, size = n)
    random_y = np.random.uniform(1000, 5000, size = n)
    random_points_A = np.array([random_y,random_x]).T

    random_x = np.random.uniform(-2500, 2000, size = n)
    random_y = np.random.uniform(1000, 5000, size = n)
    random_points_B = np.array([random_y,random_x]).T
    for i in range(n):
        loc_stat = random_points_A[i]
        loc_end = random_points_B[i]
        if f.is_path_legal(loc_stat, loc_end):
            plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="green")
        else:
            intersect_points = f.get_path_intersect(loc_stat, loc_end)
            closest_points = f.get_closest_intersect_points(loc_stat, loc_end)
            
            plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="red")
            plt.scatter(loc_stat[1], loc_stat[0], c="brown")
            plt.scatter(intersect_points[:,1], intersect_points[:,0], c="black")
            plt.scatter(closest_points[1], closest_points[0], c="blue")
        

    f.plot_operational_area()


    n = 20
    random_x = np.random.uniform(-2500, 2000, size = n)
    random_y = np.random.uniform(1000, 5000, size = n)
    random_points_A = np.array([random_y,random_x]).T

    random_x = np.random.uniform(-2500, 2000, size = n)
    random_y = np.random.uniform(1000, 5000, size = n)
    random_points_B = np.array([random_y,random_x]).T
    for i in range(n):
        loc_stat = random_points_A[i]
        loc_end = random_points_B[i]
        if f.is_path_legal(loc_stat, loc_end):
            plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="green")
        else:
            intersect_points = f.get_path_intersect(loc_stat, loc_end)
            #closest_points = f.get_closest_intersect_points(loc_stat, loc_end)
            
            plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="red")
            plt.scatter(loc_stat[1], loc_stat[0], c="brown")
            plt.scatter(intersect_points[:,1], intersect_points[:,0], c="black")
            #plt.scatter(closest_points[1], closest_points[0], c="blue")
        

    f.plot_operational_area()



