"""
Test the module WGS.py
"""

from unittest import TestCase
from WGS import WGS


class TestWGS(TestCase):

    def setUp(self) -> None:
        self.wgs = WGS()

    def test_set_origin(self) -> None:
        lat = 64
        lon = 11
        self.wgs.set_origin(lat, lon)
        x, y = self.wgs.latlon2xy(lat, lon)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_xy_conversion(self) -> None:
        x = 10
        y = 20
        lat_origin, lon_origin = self.wgs.get_origin()
        lat, lon = self.wgs.xy2latlon(x, y)
        x_new, y_new = self.wgs.latlon2xy(lat, lon)
        self.assertAlmostEquals(x, x_new)
        self.assertAlmostEquals(y, y_new)




