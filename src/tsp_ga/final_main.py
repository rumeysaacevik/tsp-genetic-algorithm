import random
import math
import time
import matplotlib.pyplot as plt
from statistics import mean


# ─────────────────── TSP Dosya Okuma ─────────────────────
def read_tsp_file(filename):
    coords = []
    with open(filename, 'r', encoding="utf-8") as file:
        first_line = file.readline().strip()
        if first_line.isdigit():
            for _ in range(int(first_line)):
                x, y = map(float, file.readline().split()[:2])
                coords.append((x, y))
        else:
            lines = [first_line] + file.readlines()
            start = False
            for line in lines:
                if 'NODE_COORD_SECTION' in line:
                    start = True
                    continue
                if not start or 'EOF' in line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords


# ───────────────── Yardımcı Hesaplamalar ─────────────────
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def total_distance(route, coords):
    return sum(
        euclidean(coords[route[i]], coords[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )


# ────────────── GA Bileşenleri ───────────────────────────
def initial_population(size, city_count):
    return [random.sample(range(city_count), city_count) for _ in range(size)]

def fitness(individual, coords):
    return 1 / total_distance(individual, coords)

def selection(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end+1] = p1[start:end+1]
    pointer = 0
    for gene in p2:
        if gene not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(individual, rate=0.02):
    for i in range(len(individual)):
        if random.random() < rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]


# ─────────────── GA Ana Fonksiyonu ───────────────────────
def genetic_algorithm(
    coords,
    population_size=150,
    generations=1000,
    mutation_rate=0.02,
    seed=None,
    log_every=100
):
    if seed:
        random.seed(seed)

    city_count = len(coords)
    population = initial_population(population_size, city_count)
    distances = [total_distance(ind, coords) for ind in population]
    best_solution = population[distances.index(min(distances))]
    best_progress = []
    t0 = time.perf_counter()

    for gen in range(generations):
        fitnesses = [1 / d for d in distances]
        new_population = []

        for _ in range(population_size):
            p1, p2 = selection(population, fitnesses)
            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        distances = [total_distance(ind, coords) for ind in population]

        current_best = population[distances.index(min(distances))]
        if total_distance(current_best, coords) < total_distance(best_solution, coords):
            best_solution = current_best

        best_progress.append(total_distance(best_solution, coords))

        if gen % log_every == 0:
            print(f"Gen {gen:>4} | En iyi mesafe: {best_progress[-1]:.2f}")

    elapsed = time.perf_counter() - t0
    return best_solution, best_progress, elapsed


# ───────────── Sonuç Çizimi ve Ana Çalıştırıcı ───────────
def plot_tour(tour, coords, title="TSP Rotası"):
    xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_progress(progress):
    plt.plot(progress)
    plt.title("GA İlerlemesi (En İyi Mesafe)")
    plt.xlabel("Jenerasyon")
    plt.ylabel("Mesafe")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tsp_path = "C:/Users/alpce/OneDrive/Belgeler/GitHub/tsp-genetic-algorithm/instances/berlin52.tsp"
    coords = read_tsp_file(tsp_path)

    best, history, runtime = genetic_algorithm(
        coords,
        population_size=150,
        generations=1000,
        mutation_rate=0.05,
        seed=42,
        log_every=50
    )

    final_dist = total_distance(best, coords)
    print(f"\nEn iyi mesafe: {final_dist:.2f} | Süre: {runtime:.2f} saniye")
    plot_tour(best, coords)
    plot_progress(history)
