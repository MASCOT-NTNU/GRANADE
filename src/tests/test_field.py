"""
Test the module Field.py
"""

from unittest import TestCase
from Field import Field
import numpy as np
import matplotlib.pyplot as plt
from numpy import testing



class TestField(TestCase):

    def setUp(self) -> None:
        self.field = Field()
        self.polygon_border = self.field.get_polygon_border()
        self.polygon_obstacle = self.field.get_polygon_obstacle()

    def show_path_polygons(self, loc_start: np.ndarray, loc_end: np.ndarray, flag: bool = False) -> None:
        if flag:
            plt.figure()
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
            plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
            plt.plot([loc_start[1], loc_end[1]], [loc_start[0], loc_end[0]], 'b-')
            plt.show()

    def test_is_loc_legal(self) -> None:
        # c1, test a point inside the border
        loc = np.array([2000, -1000])
        self.assertTrue(self.field.is_loc_legal(loc))

        # c2, test a point outside the border
        loc = np.array([2000, 1000])
        self.assertFalse(self.field.is_loc_legal(loc))

        # c3, test a point on the polygon
        ind_random = np.random.randint(0, self.polygon_border.shape[0])
        loc = self.polygon_border[ind_random, :]
        self.assertFalse(self.field.is_loc_legal(loc))

        # c4, test a point inside the obstacle
        loc = np.array([3000, -500])
        self.assertFalse(self.field.is_loc_legal(loc))

        # c5, test a point on the edge of the obstacle
        ind_random = np.random.randint(0, self.polygon_obstacle.shape[0])
        loc = self.polygon_obstacle[ind_random, :]
        self.assertFalse(self.field.is_loc_legal(loc))

    def test_is_path_legal(self) -> None:

        # c1, test a path inside the polygon border
        loc_start = np.array([1500, -1000])
        loc_end = np.array([2000, -1400])
        self.show_path_polygons(loc_start, loc_end)
        self.assertTrue(self.field.is_path_legal(loc_start,loc_end))

        # c2, test path inside the obstacle
        loc_start = np.array([3000, -800])
        loc_end = np.array([3100, -700])
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        # c3, test path colliding with border
        loc_start = np.array([1500, -1000])
        loc_end = np.array([7000, -2500])
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        # c4, test path cross the whole field
        loc_start = np.array([0, -3000])
        loc_end = np.array([7000, 3500])
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        # c5, test path colliding with obstacle
        loc_start = np.array([1500, -1000])
        loc_end = np.array([3000, -500])
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        # c6, test a path tangent to border
        ind_start = 202
        ind_end = 203
        loc_start = self.polygon_border[ind_start, :]
        loc_end = self.polygon_border[ind_end, :]
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        # c7, test a path touching the tip of the obstacle
        tip_x_coord = self.polygon_obstacle[5, :][0]
        loc_start = np.array([tip_x_coord, -1000])
        loc_end = np.array([tip_x_coord, 0])
        self.show_path_polygons(loc_start, loc_end)
        self.assertFalse(self.field.is_path_legal(loc_start, loc_end))

        
    def test_intersect_points(self) -> None:

        # c1, a path colliding with the border
        loc_start = np.array([1500, -1000])
        loc_end = np.array([7000, -2500])
        self.show_path_polygons(loc_start, loc_end, True)
        collision_point_1 = self.field.get_closest_intersect_point(loc_end , loc_start)
        collision_point_2 = self.field.get_closest_intersect_point(loc_start, loc_end)
        testing.assert_array_almost_equal(collision_point_1, collision_point_2)

        # c2, a path colliding with the obstacle 
        loc_start = np.array([2000, -800])
        loc_end = np.array([3100, -700])
        self.show_path_polygons(loc_start, loc_end, True)
        collision_point_1 = self.field.get_closest_intersect_point(loc_end , loc_start)
        collision_point_2 = self.field.get_closest_intersect_point(loc_start, loc_end)
        testing.assert_array_almost_equal(collision_point_1, collision_point_2)

        # c3, a path touching the tip of the obstacle 
        tip_point = self.polygon_obstacle[5, :]
        tip_x_coord = tip_point[0]
        loc_start = np.array([tip_x_coord, -1000])
        loc_end = np.array([tip_x_coord, 0])
        self.show_path_polygons(loc_start, loc_end, True)
        collision_point = self.field.get_closest_intersect_point(loc_start, loc_end)
        testing.assert_array_almost_equal(collision_point, tip_point)

        # c4, collision with border and obstacle, path across field
        loc_start = np.array([0, -3000])
        loc_end = np.array([7000, 3500])
        self.show_path_polygons(loc_start, loc_end, True)
        collision_point_1 = self.field.get_closest_intersect_point(loc_end , loc_start)
        collision_point_2 = self.field.get_closest_intersect_point(loc_start, loc_end)
        assert not np.array_equal(collision_point_1, collision_point_2), "The arrays should be different."

