import os
import matplotlib.pyplot as plt
from tsp_ga.ga import TSPSolverGA
from tsp_ga.tsp import read_tsp_file, euclidean_distance_matrix

def plot_route(coords, route, distance):
    route_coords = [coords[i] for i in route] + [coords[route[0]]]
    xs, ys = zip(*route_coords)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, 'o-', markersize=5)
    plt.title(f"TSP Route - Total Distance: {distance:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Dinamik path ayarı
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../instances/berlin52.tsp")
    coords = read_tsp_file(os.path.normpath(file_path))
    distance_matrix = euclidean_distance_matrix(coords)

    ga = TSPSolverGA(distance_matrix, population_size=150, generations=1000, mutation_rate=0.02, elite_size=10)
    best_route, best_distance, progress = ga.run()

    print("En kısa mesafe:", best_distance)
    print("Şehir sırası:", best_route)
    plot_route(coords, best_route, best_distance)

    # Mesafenin evrim grafiği
    plt.plot(progress)
    plt.title("Genetik Algoritma - En İyi Mesafenin Evrimi")
    plt.xlabel("Jenerasyon")
    plt.ylabel("Mesafe")
    plt.grid(True)
    plt.show()
