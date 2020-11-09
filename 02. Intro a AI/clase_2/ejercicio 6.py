import numpy as np

n = 10

def exponential_random_variable(n):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=n)
    return np.sqrt(uniform_random_variable/3)

#TODO Plot distribucion de probabilidad