import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import CsvDatasetLoader

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNNPRE", "Data")

files = CsvDatasetLoader.discover_csv_files(CSV_DIR)
print("discovered:", [f.name for f in files])
print("inferred protocols:", [CsvDatasetLoader.infer_protocol(f.name) for f in files])

loader = CsvDatasetLoader(target_dir=str(CSV_DIR))

t0 = time.perf_counter()
X, y, meta = loader.load_csv(os.path.join(CSV_DIR, "NTP1043-hex.csv"), limit=200)
dt = time.perf_counter() - t0

print(f"\nNTP load: {dt:.2f}s  X={len(X)}  y={y.shape}  classes={len(set(y))}")
print("stats:", {k: v for k, v in loader.last_stats.items() if k != "direction_map"})
print("first 3 meta:", [{k: m[k] for k in ("packet_index","protocol","direction","gt_value","gt_label","session_id","payload_length")} for m in meta[:3]])
print("payload[0] hex:", X[0].hex()[:60])

# one file per protocol, tiny limit -> check heterogeneous 'type' rendering
print("\n--- per-file probe (limit=50) ---")
for f in files:
    try:
        X2, y2, m2 = loader.load_csv(str(f), limit=50)
        vals = sorted({m["gt_value"] for m in m2})
        dirs = sorted({m["direction"] for m in m2})
        print(f"{f.name:<20} X={len(X2):>4} classes={len(set(y2)):>3} dirs={dirs} sample_types={vals[:4]}")
    except Exception as exc:
        print(f"{f.name:<20} FAILED: {type(exc).__name__}: {exc}")