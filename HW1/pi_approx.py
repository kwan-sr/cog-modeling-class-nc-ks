import random
import numpy as np

# Change number of random generations here
# the larger N is, the more accurate the approx.
N = 1000000

# Generate N random points as (x,y) tuples
# each x and y is in range [-1,1]
random_points = [(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for _ in range(N)]

# This extracts just the coordinates that's inside the unit circle
# which means x^2 + y^2 <= 1
in_points = [point for point in random_points if (point[0]**2 + point[1]**2 <= 1)]

# Approximate pi by the amount of dots inside the circle compared to total dots
approx_pi = 4 * len(in_points) / N

# Report estimated pi value
print(f"According to this Monte Carlo simulation with N={N}, pi is approximately {approx_pi}")