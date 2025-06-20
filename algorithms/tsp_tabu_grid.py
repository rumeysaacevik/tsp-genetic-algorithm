# tsp_tabu_grid.py
# ------------------------------------------------------------
# Tabu Search (2-opt + grid search) ile TSP çözümlerini test eder ve CSV'ye kaydeder
# Temizlenmiş, yeniden adlandırılmış, yorumlanmış versiyon
# ------------------------------------------------------------

import math
import random
import time
import itertools
import csv
from collections import deque
from statistics import mean
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------- Veri Okuma ----------------------------
def read_tsp(path):
    coords = []
    with open(path, 'r', encoding='utf-8') as file:
        first = file.readline().strip()
        if first.isdigit():
            for _ in range(int(first)):
                x, y = map(float, file.readline().split()[:2])
                coords.append((x, y))
        else:
            lines = [first] + file.readlines()
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
    if not coords:
        raise ValueError("Koordinatlar bulunamadı veya dosya hatalı.")
    return coords

# ---------------------------- Yardımcı Fonksiyonlar ----------------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def tour_length(tour, cities):
    return sum(euclidean(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def two_opt(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]

def nearest_start(cities):
    n = len(cities)
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda j: euclidean(cities[last], cities[j]))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

# ---------------------------- Tabu Search ----------------------------
def tabu_search(cities, tabu_size=50, max_iter=10000, neighbor_sample=200, aspiration=True, seed=None, log_step=500):
    if seed is not None:
        random.seed(seed)

    n = len(cities)
    current = nearest_start(cities)
    best = current[:]
    best_dist = tour_length(best, cities)
    tabu = deque(maxlen=tabu_size)

    start_time = time.perf_counter()
    for iter in range(1, max_iter + 1):
        best_candidate = None
        best_candidate_len = float('inf')
        move = None

        for _ in range(neighbor_sample):
            i, k = sorted(random.sample(range(1, n), 2))
            if (i, k) in tabu and not aspiration:
                continue
            neighbor = two_opt(current, i, k)
            dist = tour_length(neighbor, cities)
            if dist < best_candidate_len or (aspiration and dist < best_dist and (i, k) in tabu):
                best_candidate = neighbor
                best_candidate_len = dist
                move = (i, k)

        if best_candidate is None:
            break

        current = best_candidate
        tabu.append(move)

        if best_candidate_len < best_dist:
            best = best_candidate[:]
            best_dist = best_candidate_len

        if iter % log_step == 0:
            print(f"Iter {iter:>5} | Best = {best_dist:.2f} | Current = {best_candidate_len:.2f}")

    return best, time.perf_counter() - start_time

# ---------------------------- Parametre Tarama ----------------------------
GRID = {
    "neighbor_sample": [50, 100, 150],
    "max_iter": [300, 600],
    "tabu_size": [1, 5]
}
REPEATS = 5
SEED_BASE = 2025
TSP_FILE = "../data/tsp_100_1"
CSV_FILE = "tabu_search_results.csv"

def run_grid():
    cities = read_tsp(TSP_FILE)
    results = []
    for ns, mi, ts in itertools.product(GRID["neighbor_sample"], GRID["max_iter"], GRID["tabu_size"]):
        print(f"\nTesting → neighbor_sample={ns}, max_iter={mi}, tabu_size={ts}")
        dists, times = [], []
        for r in range(REPEATS):
            seed = SEED_BASE + r
            sol, duration = tabu_search(cities, tabu_size=ts, max_iter=mi, neighbor_sample=ns, seed=seed, log_step=mi+1)
            dist = tour_length(sol, cities)
            dists.append(dist)
            times.append(duration)
            print(f"  ▸ Repeat {r+1}/{REPEATS}: dist={dist:.2f}, time={duration:.2f}s")

        results.append({
            "neighbor_sample": ns,
            "max_iter": mi,
            "tabu_size": ts,
            "best_dist": min(dists),
            "mean_dist": mean(dists),
            "std_dist": (sum((d - mean(dists))**2 for d in dists) / (REPEATS - 1))**0.5,
            "mean_time": mean(times),
        })
    return results

def save_csv(rows, path=CSV_FILE):
    fields = ["neighbor_sample", "max_iter", "tabu_size", "best_dist", "mean_dist", "std_dist", "mean_time"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSonuclar '{path}' dosyasina kaydedildi.")

# ---------------------------- Ana Çalıştırıcı ----------------------------
if __name__ == "__main__":
    results = run_grid()
    save_csv(results)
