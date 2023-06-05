import numpy as np

@staticmethod
def sliding_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size