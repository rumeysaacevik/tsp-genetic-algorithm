import csv
import time
import itertools
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Gerekli fonksiyonları içe aktar (senin fonksiyon isimlerine göre)
from algorithms.tsp_ga_baseline import (
    load_tsp_file,
    tour_length,
    run_genetic_algorithm,
)

# Parametre kombinasyonları
GRID_PARAMS = {
    "pop_size":      [50, 100, 150],
    "max_gen":       [300, 600],
    "mutation_rate": [0.01, 0.05],
}

REPEAT_EACH = 3
SEED_BASE   = 1234
OUT_CSV     = "ga_experiments.csv"

# Deneyleri çalıştır
def execute_trials(cities):
    keys, values = zip(*GRID_PARAMS.items())
    scenarios = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    results = []
    total_runs = len(scenarios) * REPEAT_EACH
    run_count = 0

    for config in scenarios:
        for repeat in range(REPEAT_EACH):
            run_count += 1
            current_seed = SEED_BASE + run_count
            print(f"[{run_count:>3}/{total_runs}] {config} tekrar={repeat+1}")

            best_route, duration = run_genetic_algorithm(
                cities,
                pop_size      = config["pop_size"],
                max_gen       = config["max_gen"],
                mutation_rate = config["mutation_rate"],
                seed          = current_seed,
                log_freq      = config["max_gen"] + 1  # sessiz
            )

            best_len = tour_length(best_route, cities)

            results.append({
                **config,
                "repeat":        repeat + 1,
                "best_distance": best_len,
                "exec_time":     duration,
            })

    return results

# CSV yaz
def dump_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Sonuçlar CSV olarak kaydedildi: {filename}")

# Özet tablo ve grafik
def summarize(results):
    df = pd.DataFrame(results)
    agg = (
        df.groupby(["pop_size", "max_gen", "mutation_rate"])
          .agg(best_mean=("best_distance", "mean"),
               best_std=("best_distance", "std"),
               time_mean=("exec_time", "mean"))
          .reset_index()
    )

    print("\n📊 Deney Özeti:")
    print(agg.to_string(index=False, formatters={
        "best_mean": "{:.2f}".format,
        "best_std":  "{:.2f}".format,
        "time_mean": "{:.2f}".format,
    }))

    fig, ax = plt.subplots()
    ax.scatter(agg["time_mean"], agg["best_mean"])
    for _, row in agg.iterrows():
        label = f"P{row.pop_size}|G{row.max_gen}|M{row.mutation_rate}"
        ax.annotate(label, (row.time_mean, row.best_mean))
    ax.set_xlabel("Ortalama Süre (sn)")
    ax.set_ylabel("Ortalama En İyi Mesafe")
    ax.set_title("GA Parametre Deneyi")
    plt.tight_layout()
    plt.show()

# Ana fonksiyon
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=OUT_CSV,
                        help=f"Çıktı CSV dosyası (varsayılan: {OUT_CSV})")
    args = parser.parse_args()

    tsp_path = "../data/tsp_100_1"
    if not Path(tsp_path).exists():
        parser.error(f"Girdi dosyası bulunamadı: {tsp_path}")

    cities = load_tsp_file(tsp_path)
    start = time.perf_counter()
    all_results = execute_trials(cities)
    dump_to_csv(all_results, args.csv)
    summarize(all_results)
    print(f"\n⏱️ Toplam deney süresi: {time.perf_counter() - start:.1f} saniye")

if __name__ == "__main__":
    main()
