import csv
import time
import itertools
import argparse
from pathlib import Path
import pandas as pd

from algorithms.tsp_ga_adaptive import solve_tsp_ga


# ----------------- Parametre Alanı -----------------
CONFIGS = {
    "size":    [50, 100, 150],
    "epochs":  [300, 600],
    "mutate":  [0.10, 0.20],
}

N_REPEAT   = 3
SEED_START = 2024
OUTFILE    = "ga_results_70.csv"


# ----------------- Deney Fonksiyonu -----------------
def perform_tests(data_path):
    setting_list = list(itertools.product(*CONFIGS.values()))
    param_keys   = list(CONFIGS.keys())
    all_logs     = []
    total_jobs   = len(setting_list) * N_REPEAT
    counter      = 0

    for config in setting_list:
        params = dict(zip(param_keys, config))
        for run in range(N_REPEAT):
            counter += 1
            seed_val = SEED_START + counter
            print(f"[{counter:>3}/{total_jobs}] Params: {params} | rep={run+1}")

            result, duration = solve_tsp_ga(
                data_path,
                pop_size=params["size"],
                max_gen=params["epochs"],
                mutation_base=params["mutate"],
                elite_count=1,
                seed=seed_val,
                verbose=False,
                log_interval=params["epochs"] + 1,
            )

            all_logs.append({
                "pop_size":       params["size"],
                "max_gen":        params["epochs"],
                "mutation_base":  params["mutate"],
                "repeat":         run + 1,
                "best_distance":  result,
                "exec_time":      duration,
            })

    return all_logs


# ----------------- CSV Kaydetme -----------------
def export_csv(logs, path):
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    print(f"\n📁 CSV dosyası yazıldı: {path}")


# ----------------- Özet Tablo -----------------
def summarize_results(log_data):
    df = pd.DataFrame(log_data)
    summary = (
        df.groupby(["pop_size", "max_gen", "mutation_base"])
          .agg(best_mean=("best_distance", "mean"),
               best_std=("best_distance", "std"),
               time_mean=("exec_time", "mean"))
          .reset_index()
    )

    print("\n📊 SONUÇ ÖZETİ")
    print(summary.to_string(index=False, formatters={
        "best_mean": "{:.1f}".format,
        "best_std":  "{:.1f}".format,
        "time_mean": "{:.2f}".format,
    }))


# ----------------- Ana Fonksiyon -----------------
def main():
    parser = argparse.ArgumentParser(description="Memetik GA grid search")
    file_path = "../data/tsp_70_1"
    parser.add_argument("--csv", default=OUTFILE,
                        help=f"Çıktı CSV dosyası (default: {OUTFILE})")
    args = parser.parse_args()

    if not Path(file_path).exists():
        parser.error("Veri dosyası bulunamadı!")

    t_start = time.perf_counter()
    log_data = perform_tests(file_path)
    export_csv(log_data, args.csv)
    summarize_results(log_data)
    print(f"\n⏱️ Toplam süre: {time.perf_counter() - t_start:.1f} saniye")


if __name__ == "__main__":
    main()
