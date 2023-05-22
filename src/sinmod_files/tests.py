
import numpy as np
from scipy.spatial import distance_matrix


a = np.arange(8)
b = np.arange(8)

print("a", a)
print("b", b)

M = distance_matrix(a.reshape(-1,1), b.reshape(-1,1))

print("dist matrix \n", M)

a = np.random.rand(800)
b = np.random.rand(800)
zero_vec = np.zeros(800)
print(np.array([a, zero_vec]).shape)    

M = distance_matrix(np.array([a, zero_vec]).T, np.array([b, zero_vec]).T)

print("dist matrix \n", M.shape)

print(a.reshape(-1,1).shape)