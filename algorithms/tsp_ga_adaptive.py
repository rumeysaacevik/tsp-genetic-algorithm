# tsp_ga_adaptive.py
# ------------------------------------------------------------
# Genetik Algoritma (PMX + 2-opt + Adaptif Mutasyon) ile TSP Çözümü
# Temiz, açıklamalı, yeniden yazılmış versiyon
# ------------------------------------------------------------
from __future__ import annotations

import sys
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

def print_log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)

def solve_tsp_ga(
    filepath: str = "input.txt",
    *,
    pop_size: int = 50,
    max_gen: int = 500,
    mutation_base: float = 0.20,
    elite_count: int = 1,
    seed: int | None = None,
    verbose: bool = True,
    log_interval: int = 10,
) -> Tuple[float, float]:
    if seed is not None:
        random.seed(seed)

    start_time = time.perf_counter()

    # -------------------- Dosya ve Koordinat Okuma --------------------
    file = Path(filepath)
    if not file.exists():
        raise FileNotFoundError(filepath)

    print_log(f"[1/8] Veri yükleniyor: '{filepath}'", verbose)
    data = file.read_text().strip().split()
    if not data:
        raise ValueError("Dosya boş görünüyor.")

    N = int(data[0])
    expected_length = 1 + 2 * N
    if len(data) < expected_length:
        raise ValueError(f"{N} şehir için {2*N} koordinat gerekir, ancak {len(data)-1} bulundu.")

    coords = list(map(float, data[1:expected_length]))
    cities: List[Tuple[float, float]] = list(zip(coords[::2], coords[1::2]))
    print_log(f"--> {N} şehir başarıyla okundu.", verbose)

    # -------------------- Mesafe Matrisi --------------------
    print_log("[2/8] Mesafeler hesaplanıyor...", verbose)
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        x1, y1 = cities[i]
        for j in range(i + 1, N):
            x2, y2 = cities[j]
            d = int(math.floor(math.hypot(x1 - x2, y1 - y2) + 0.5))
            dist[i][j] = dist[j][i] = d
    print_log("--> Mesafe matrisi tamam.", verbose)

    # -------------------- Popülasyon --------------------
    def nearest_neighbor(start: int = 0) -> List[int]:
        tour = [start]
        unvisited = set(range(N)) - {start}
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda j: dist[last][j])
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    population: List[List[int]] = [nearest_neighbor()]
    while len(population) < pop_size:
        route = list(range(N))
        random.shuffle(route)
        population.append(route)
    print_log(f"--> Popülasyon oluşturuldu: {pop_size} birey", verbose)

    # -------------------- GA Yardımcı Fonksiyonlar --------------------
    def route_distance(tour: List[int]) -> int:
        return sum(dist[tour[i]][tour[(i + 1) % N]] for i in range(N))

    def tournament_selection() -> int:
        candidates = random.sample(range(len(population)), 3)
        return min(candidates, key=lambda idx: fitness[idx])

    def pmx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        a, b = sorted(random.sample(range(N), 2))
        c1, c2 = [-1] * N, [-1] * N
        c1[a:b+1] = p1[a:b+1]
        c2[a:b+1] = p2[a:b+1]

        def fill(child, donor):
            for i in range(a, b + 1):
                gene = donor[i]
                if gene not in child:
                    pos = i
                    while child[pos] != -1:
                        pos = donor.index(child[pos])
                    child[pos] = gene

        fill(c1, p2)
        fill(c2, p1)

        for i in range(N):
            if c1[i] == -1:
                c1[i] = p2[i]
            if c2[i] == -1:
                c2[i] = p1[i]

        return c1, c2

    def swap_mutation(route: List[int]) -> None:
        i, j = random.sample(range(N), 2)
        route[i], route[j] = route[j], route[i]

    def two_opt(route: List[int]) -> int:
        improved = True
        total = route_distance(route)
        while improved:
            improved = False
            for i in range(1, N - 2):
                for j in range(i + 2, N):
                    a, b = route[i - 1], route[i]
                    c, d = route[j], route[(j + 1) % N]
                    old_cost = dist[a][b] + dist[c][d]
                    new_cost = dist[a][c] + dist[b][d]
                    if new_cost < old_cost:
                        route[i:j+1] = reversed(route[i:j+1])
                        total += new_cost - old_cost
                        improved = True
                        break
                if improved:
                    break
        return total

    # -------------------- GA Başlangıç Değerlendirme --------------------
    fitness = [1.0 / route_distance(ind) for ind in population]
    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    best_route = population[best_idx][:]
    best_score = 1.0 / fitness[best_idx]
    print_log(f"[4/8] İlk en iyi skor: {best_score}", verbose)

    # -------------------- GA Döngüsü --------------------
    mutation_rate = mutation_base
    no_improvement = 0

    for gen in range(1, max_gen + 1):
        next_gen, next_fit = [], []

        # Elitizm
        for _ in range(elite_count):
            next_gen.append(best_route[:])
            next_fit.append(1.0 / best_score)

        while len(next_gen) < pop_size:
            p1, p2 = population[tournament_selection()], population[tournament_selection()]
            c1, c2 = pmx_crossover(p1, p2)
            if random.random() < mutation_rate:
                swap_mutation(c1)
            if random.random() < mutation_rate:
                swap_mutation(c2)
            d1, d2 = two_opt(c1), two_opt(c2)
            next_gen.extend([c1, c2])
            next_fit.extend([1.0 / d1, 1.0 / d2])

        population, fitness = next_gen[:pop_size], next_fit[:pop_size]
        current_idx = max(range(pop_size), key=lambda i: fitness[i])
        current_score = 1.0 / fitness[current_idx]

        if current_score < best_score:
            best_score = current_score
            best_route = population[current_idx][:]
            mutation_rate = mutation_base
            no_improvement = 0
            print_log(f"[Gen {gen}] Gelişme! Yeni en iyi: {best_score}", verbose)
        else:
            no_improvement += 1
            if no_improvement == 5:
                mutation_rate = min(1.0, mutation_base * 2)
            elif no_improvement == 20:
                mutation_rate = min(1.0, mutation_base * 4)

        if gen % log_interval == 0:
            print_log(f"[Gen {gen}] En iyi: {best_score}", verbose)

    elapsed = time.perf_counter() - start_time
    print(f"{best_score} 0")
    print(" ".join(map(str, best_route)))
    print_log(f"Toplam süre: {elapsed:.2f} sn", verbose)
    return best_score, elapsed

if __name__ == "__main__":
    solve_tsp_ga(
        "../data/tsp_100_1",
        seed=42  # ⬅ sabitlik için bunu ekledik
    )

