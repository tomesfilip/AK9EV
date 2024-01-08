import numpy as np


def check_bounds(position, min_b, max_b):
    reflected = np.where(position < min_b, 2 * min_b - position, position)
    reflected = np.where(reflected > max_b, 2 * max_b - reflected, reflected)
    return reflected
