import numpy as np

def read_tsp_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))
            except:
                continue
    return coords

def euclidean_distance_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                matrix[i][j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    return matrix
