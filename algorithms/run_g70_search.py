import argparse, math, random, sys, time
from pathlib import Path
from typing import List, Tuple

def _y(p: Path) -> List[Tuple[float, float]]:
    l = p.read_text().strip().split()
    k = int(l[0])
    d = list(zip(map(float, l[1::2]), map(float, l[2::2])))[:k]
    if len(d) != k:
        raise ValueError(f"{p}: expected {k} coords, got {len(d)}")
    return d

def _d(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _z(t: List[int], u: List[Tuple[float,float]]) -> float:
    return sum(_d(u[t[i]], u[t[(i+1)%len(t)]]) for i in range(len(t)))

def _x(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1); a, b = sorted(random.sample(range(n), 2))
    c = [-1]*n; c[a:b] = p1[a:b]
    q = 0
    for g in p2:
        if g not in c:
            while c[q] != -1: q += 1
            c[q] = g
    return c

def _m(t: List[int], r: float):
    if random.random() < r:
        i, j = random.sample(range(len(t)), 2)
        t[i], t[j] = t[j], t[i]

def _t(t, u):
    f = True
    while f:
        f = False
        for i in range(1, len(t)-2):
            for j in range(i+1, len(t)):
                if j - i == 1: continue
                a, b = t[i-1], t[i]
                c, d = t[j-1], t[j%len(t)]
                if _d(u[a], u[b]) + _d(u[c], u[d]) > _d(u[a], u[c]) + _d(u[b], u[d]):
                    t[i:j] = reversed(t[i:j])
                    f = True

def _g(a, b=200, c=3000, d=0.02, e=False, f=0):
    v = a
    n = len(v)
    P = [random.sample(range(n), n) for _ in range(b)]
    F = [_z(t, v) for t in P]
    for g in range(c):
        if f and (g % f == 0 or g == c-1):
            print(f"Gen {g:4d}: best={min(F):.2f}  avg={sum(F)/b:.2f}  worst={max(F):.2f}")
        h = min(range(b), key=F.__getitem__)
        s = P[h][:]
        W = [s]
        while len(W) < b:
            u, w = random.sample(P, 2)
            q = _x(u, w)
            _m(q, d)
            if e: _t(q, v)
            W.append(q)
        P = W
        F = [_z(t, v) for t in P]
    h = min(range(b), key=F.__getitem__)
    return P[h], F[h]

def _h():
    a = argparse.ArgumentParser()
    a.add_argument("file", type=Path)
    a.add_argument("--pop", type=int, default=200)
    a.add_argument("--gens", type=int, default=3000)
    a.add_argument("--mut", type=float, default=0.02)
    a.add_argument("--twoopt", action="store_true")
    a.add_argument("--log", type=int, default=0)
    a.add_argument("--seed", type=int)
    c = a.parse_args()
    if c.seed is not None:
        random.seed(c.seed)
    b = _y(c.file)
    s = time.perf_counter()
    t, best = _g(b, c.pop, c.gens, c.mut, c.twoopt, c.log)
    print(f"\nFinal best length = {best:.3f}  (time {time.perf_counter()-s:.2f}s)")


def _q(
    a: str,
    b: dict,
    r: int = 1,
    p: str = "../experiment_results/ga4_experiments_70.csv",
) -> None:
    c = _y(Path(a))
    t = [(x, y, z) for x in b["population_size"] for y in b["generations"] for z in b["mutation_rate"]]
    rows = [("pop", "gens", "mut", "run", "best_len", "seconds")]
    i = 1
    print(f"Running {len(t)*r} GA runs …")
    for pop, gens, mut in t:
        for run in range(1, r+1):
            ts = time.perf_counter()
            _, best = _g(c, pop, gens, mut, e=True, f=0)
            dt = time.perf_counter() - ts
            rows.append((pop, gens, mut, run, round(best, 3), round(dt, 2)))
            print(f"[{i:>3}/{len(t)*r}] pop={pop:<3} gens={gens:<4} mut={mut:<4} -> best={best:.3f}  ({dt:.2f}s)")
            i += 1
    with open(p, "w", newline="") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"\nResults written to {p}")


J = {
    "population_size": [50, 100, 150],
    "generations": [300, 600],
    "mutation_rate": [0.01, 0.05],
}

if __name__ == "__main__":
    _q("../data/tsp_70_1", J, 3, "../experiment_results/ga4_experiments_70.csv")
