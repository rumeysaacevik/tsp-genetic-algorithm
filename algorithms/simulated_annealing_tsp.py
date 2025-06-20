"""
SA tabanlı TSP çözücü - yapılandırılmış
"""
import math, random, time
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

# --- Dosya çözüm fonksiyonu ---
def parse_input(f):
    points = []
    with open(f, "r", encoding="utf-8") as src:
        head = src.readline().strip()
        if head.isdigit():
            for _ in range(int(head)):
                x, y = map(float, src.readline().split()[:2])
                points.append((x, y))
        else:
            all_lines = [head] + src.readlines()
            activate = False
            for l in all_lines:
                if "NODE_COORD_SECTION" in l:
                    activate = True
                    continue
                if not activate or "EOF" in l:
                    continue
                vals = l.split()
                if len(vals) >= 3:
                    points.append((float(vals[1]), float(vals[2])))
    if not points:
        raise RuntimeError("Koordinat verisi okunamıyor.")
    return points

# --- Yardımcılar ---
def calc(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def path_sum(route, pts):
    return sum(calc(pts[route[i]], pts[route[(i+1)%len(route)]]) for i in range(len(route)))

def flip(route, a, b):
    return route[:a] + route[a:b+1][::-1] + route[b+1:]

# --- SA motoru ---
def anneal(pts, init_temp=1e4, decay=0.995, min_temp=1e-3, limit=1e5, seed=42, gap=1000):
    if seed is not None:
        random.seed(seed)

    n = len(pts)
    walk = list(range(n))
    record = walk[:]
    best_val = path_sum(record, pts)

    T = init_temp
    t0 = time.perf_counter()
    history = []

    for it in range(1, int(limit) + 1):
        i, j = sorted(random.sample(range(1, n), 2))
        trial = flip(walk, i, j)
        delta = path_sum(trial, pts) - path_sum(walk, pts)

        if delta < 0 or math.exp(-delta / T) > random.random():
            walk = trial

        score = path_sum(walk, pts)
        if score < best_val:
            record, best_val = walk[:], score

        history.append(score)

        if it % gap == 0:
            print(f"Adım {it:>6} | T={T:7.2f} | En İyi={best_val:8.2f} | Ort={mean(history[-gap:]):8.2f} | Anlık={score:8.2f}")

        T *= decay
        if T < min_temp:
            break

    total_t = time.perf_counter() - t0
    print(f"\nTamamlandı. Süre: {total_t:.2f} sn")
    return record, total_t

# --- Grafik ---
def draw(route, pts, label="SA ile bulunan rota"):
    xs = [pts[i][0] for i in route] + [pts[route[0]][0]]
    ys = [pts[i][1] for i in route] + [pts[route[0]][1]]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.title(label)
    plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Ana blok ---
if __name__ == "__main__":
    dosya = "../data/tsp_100_1"
    noktalar = parse_input(dosya)
    print(f"{len(noktalar)} nokta yüklendi. SA başlatılıyor...")
    sonuc, gecen = anneal(noktalar, init_temp=10000, decay=0.995, min_temp=1e-3, limit=50000, seed=42, gap=1000)
    uzunluk = path_sum(sonuc, noktalar)
    print(f"En kısa mesafe: {uzunluk:.2f}")
    draw(sonuc, noktalar)
