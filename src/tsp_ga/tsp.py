# tsp.py
import math

def read_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().isdigit():
                continue
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                coords.append((float(parts[1]), float(parts[2])))
    return coords

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def calculate_distance_matrix(coords):
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = euclidean_distance(coords[i], coords[j])
    return matrix
