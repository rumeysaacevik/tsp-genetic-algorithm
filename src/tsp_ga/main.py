from tsp import read_tsp_file, calculate_distance_matrix
from ga import GeneticAlgorithm

if __name__ == "__main__":
    coords = read_tsp_file("../../instances/berlin52.tsp")
    dist_matrix = calculate_distance_matrix(coords)
    ga = GeneticAlgorithm(dist_matrix)
    best_tour, best_distance = ga.run()

    print("Best Tour:", best_tour)
    print("Total Distance:", best_distance)
