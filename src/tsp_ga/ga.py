import random

def create_individual(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

def evaluate(individual, distance_matrix):
    total = 0
    for i in range(len(individual)):
        a = individual[i]
        b = individual[(i + 1) % len(individual)]  # Loop back to start
        total += distance_matrix[a][b]
    return total

class GeneticAlgorithm:
    def __init__(self, distance_matrix, pop_size=100, generations=500, crossover_rate=0.9, mutation_rate=0.02):
        self.matrix = distance_matrix
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_cities = len(distance_matrix)

        # Popülasyonu başlat
        self.population = [create_individual(self.num_cities) for _ in range(self.pop_size)]


    def run(self):
        for gen in range(self.generations):
            self.population.sort(key=lambda x: evaluate(x, self.matrix))
            next_gen = self.population[:10]  # Elitism: En iyi 10 bireyi koru

            while len(next_gen) < self.pop_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1[:]
                if random.random() < self.mutation_rate:
                    self.swap_mutation(child)
                next_gen.append(child)

            self.population = next_gen

        best = min(self.population, key=lambda x: evaluate(x, self.matrix))
        return best, evaluate(best, self.matrix)

    def tournament_selection(self, k=5):
        selected = random.sample(self.population, k)
        selected.sort(key=lambda x: evaluate(x, self.matrix))
        return selected[0]

    def order_crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end+1] = parent1[start:end+1]
        p2_index = 0
        for i in range(len(parent1)):
            if child[i] is None:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]
        return child

    def swap_mutation(self, individual):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
