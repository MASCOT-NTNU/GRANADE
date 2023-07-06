import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from WGS import WGS


border = pd.read_csv(os.getcwd() + "/src/csv/polygon_border.csv").to_numpy()
border_small = pd.read_csv(os.getcwd() + "/src/csv/polygon_border_small.csv").to_numpy()
obstacles = pd.read_csv(os.getcwd() + "/src/csv/polygon_obstacle.csv").to_numpy()
safety_zone = pd.read_csv(os.getcwd() + "/src/csv/polygon_safety.csv").to_numpy()

print(border.shape)


plt.plot(obstacles[:,1], obstacles[:,0])
plt.plot(border[:,1], border[:,0])
plt.plot(border_small[:,1], border_small[:,0])
plt.plot(safety_zone[:,1], safety_zone[:,0])
plt.scatter(border_small[:,1], border_small[:,0], c=np.linspace(0,border_small.shape[0],border_small.shape[0]) , cmap='turbo')
plt.colorbar()
plt.show()

# Trying to find the corner points of the polygo

interesting_points = [201, 202,203, 80]

plt.plot(border[:,1], border[:,0])
plt.plot(obstacles[:,1], obstacles[:,0])
for i in interesting_points:
    plt.scatter(border[i,1], border[i,0], c="r", label=i)
plt.legend()
plt.show()

# Finding the middle point between two points
def find_middle_point(p1, p2):
    return [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]

plt.plot(border[:,1], border[:,0])
plt.plot(obstacles[:,1], obstacles[:,0])
for i in interesting_points:
    plt.scatter(border[i,1], border[i,0], c="r", label=i)
mid_point = find_middle_point(border[202], border[203])
print(mid_point)
plt.scatter(mid_point[1], mid_point[0], c="g", label="mid point")
plt.legend()
plt.show()


x,y = WGS.latlon2xy(border_small[:,0], border_small[:,1])
x_b, y_b = WGS.latlon2xy(border[:,0], border[:,1])
x_o, y_o = WGS.latlon2xy(obstacles[:,0], obstacles[:,1])
plt.plot(y_b, x_b)
plt.plot(y_o, x_o)
plt.plot(y,x)


plt.show()
