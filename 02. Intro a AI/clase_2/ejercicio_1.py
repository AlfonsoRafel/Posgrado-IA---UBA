import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
C = np.array([[1, 0, 0], [0, 1, 1]])

# Ayuda a conseguir las distancias sin un for
expanded_C = C[:, None]
distances = np.sqrt(np.sum((expanded_C-X)**2, axis=2))

print(distances)
