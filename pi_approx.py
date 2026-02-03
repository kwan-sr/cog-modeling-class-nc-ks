import random
import numpy as np

N = 100
random_points = random_uniform_floats = [(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for _ in range(N)]
in_points = [point for point in random_points if (point[1]**2 + point[2]**2)]

print(random_points)