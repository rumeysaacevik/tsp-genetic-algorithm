import os
import random
import math
import time
from statistics import mean
import matplotlib.pyplot as plt

# ----------------------- TSP Veri Okuma -----------------------
def load_tsp_file(path):
    """TSP dosyasını oku, koordinatları liste olarak döndür."""
    coordinates = []
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip()
        if first.isdigit():
            for _ in range(int(first)):
                x, y = map(float, f.readline().split())
                coordinates.append((x, y))
        else:
            lines = [first] + f.readlines()
            reading = False
            for line in lines:
                if 'NODE_COORD_SECTION' in line:
                    reading = True
                    continue
                if 'EOF' in line or not reading:
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    coordinates.append((float(parts[1]), float(parts[2])))
    if not coordinates:
        raise ValueError(f"Veri okunamadı veya format hatalı: {path}")
    return coordinates

# ----------------------- Mesafe Fonksiyonları -----------------------
def calc_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def tour_length(route, cities):
    return sum(
        calc_distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )

# ----------------------- GA Yardımcı Fonksiyonlar -----------------------
def create_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def evaluate_fitness(route, cities):
    return 1.0 / tour_length(route, cities)

def select_parents(population, fitness_scores):
    return random.choices(population, weights=fitness_scores, k=2)

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]

    pos = 0
    for gene in parent2:
        if gene not in child:
            while child[pos] != -1:
                pos += 1
            child[pos] = gene
    return child

def apply_mutation(route, rate=0.01):
    for i in range(len(route)):
        if random.random() < rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]

# ----------------------- Genetik Algoritma Ana Fonksiyonu -----------------------
def run_genetic_algorithm(cities, pop_size=100, max_gen=500, mutation_rate=0.01, seed=None, log_freq=1):
    if seed is not None:
        random.seed(seed)

    start_time = time.perf_counter()
    population = create_population(pop_size, len(cities))
    distances = [tour_length(ind, cities) for ind in population]
    best = population[distances.index(min(distances))]

    for gen in range(max_gen):
        fitnesses = [1.0 / d for d in distances]
        new_gen = []

        for _ in range(pop_size):
            p1, p2 = select_parents(population, fitnesses)
            offspring = ordered_crossover(p1, p2)
            apply_mutation(offspring, mutation_rate)
            new_gen.append(offspring)

        population = new_gen
        distances = [tour_length(ind, cities) for ind in population]
        current_best = population[distances.index(min(distances))]

        if tour_length(current_best, cities) < tour_length(best, cities):
            best = current_best

        if gen % log_freq == 0:
            print(f"Gen {gen:>3} | En İyi: {min(distances):.2f} | Ortalama: {mean(distances):.2f}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTamamlandı ⏱️ Süre: {elapsed:.2f} sn")
    return best, elapsed

# ----------------------- Sonucu Görselleştir -----------------------
def draw_route(route, cities, title="Optimal TSP Rotası"):
    xs = [cities[i][0] for i in route] + [cities[route[0]][0]]
    ys = [cities[i][1] for i in route] + [cities[route[0]][1]]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("X Koordinat")
    plt.ylabel("Y Koordinat")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------- Uygulama Başlat -----------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    tsp_path = os.path.join(BASE_DIR, "../data/tsp_100_1")
    city_list = load_tsp_file(tsp_path)

    best_tour, runtime = run_genetic_algorithm(
        city_list,
        pop_size=150,
        max_gen=1000,
        mutation_rate=0.01,
        seed=7,
        log_freq=10
    )

    print(f"En kısa yol uzunluğu: {tour_length(best_tour, city_list):.2f}")
    draw_route(best_tour, city_list)