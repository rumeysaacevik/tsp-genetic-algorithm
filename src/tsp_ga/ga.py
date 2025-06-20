import random
import numpy as np

class TSPSolverGA:
    def __init__(self, distance_matrix, population_size=100, generations=500, mutation_rate=0.01, elite_size=5):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def initial_population(self):
        return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]

    def fitness(self, route):
        return 1 / self.route_distance(route)

    def route_distance(self, route):
        distance = sum(self.distance_matrix[route[i]][route[i+1]] for i in range(self.num_cities - 1))
        distance += self.distance_matrix[route[-1]][route[0]]
        return distance

    def selection(self, population):
        fitness_scores = [(ind, self.fitness(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [ind for ind, _ in fitness_scores[:self.elite_size]]
        while len(selected) < self.population_size:
            selected.append(self.tournament(population))
        return selected

    def tournament(self, population, k=3):
        selected = random.sample(population, k)
        selected.sort(key=lambda x: self.fitness(x), reverse=True)
        return selected[0]

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [-1] * self.num_cities
        child[start:end] = parent1[start:end]
        fill = [city for city in parent2 if city not in child]
        pointer = 0
        for i in range(self.num_cities):
            if child[i] == -1:
                child[i] = fill[pointer]
                pointer += 1
        return child

    def mutate(self, individual):
        for i in range(self.num_cities):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.num_cities - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve(self, population):
        selected = self.selection(population)
        children = selected[:self.elite_size]
        while len(children) < self.population_size:
            p1, p2 = random.sample(selected, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            children.append(child)
        return children

    def run(self):
        population = self.initial_population()
        best_route = None
        best_distance = float("inf")
        progress = []

        for gen in range(self.generations):
            population = self.evolve(population)
            current_best = min(population, key=self.route_distance)
            current_distance = self.route_distance(current_best)

            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_best

            progress.append(best_distance)

            # Her 100 jenerasyonda bir çıktı verelim
            if gen % 100 == 0 or gen == self.generations - 1:
                print(f"Jenerasyon {gen}: En iyi mesafe = {current_distance:.2f}")

        return best_route, best_distance, progress

