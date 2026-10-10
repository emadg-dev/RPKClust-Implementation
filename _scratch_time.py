import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import CsvDatasetLoader
from rpkclust import RPKClust

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNNPRE", "Data")
META_FIELDS = ("timestamp","protocol","transport","direction","source_ip","source_port",
               "destination_ip","destination_port","session_id","packet_index","payload_length")

def project(meta):
    return [{k: r[k] for k in META_FIELDS if k in r} for r in meta]

loader = CsvDatasetLoader(target_dir=str(CSV_DIR))

for name in ("NTP1043-hex.csv", "HTTP1963-hex.csv"):
    path = os.path.join(CSV_DIR, name)
    for n in (50, 100, 200, 400):
        X, y, meta = loader.load_csv(path, limit=n)
        fm = project(meta)
        ts = [r["timestamp"] for r in meta if r.get("timestamp") is not None]
        kw = dict(interaction_metadata=fm, direction_labels=[r.get("direction","unknown") for r in meta])
        if ts:
            kw["capture_start"], kw["capture_end"] = min(ts), max(ts)
        t0 = time.perf_counter()
        m = RPKClust().fit(X, **kw)
        dt = time.perf_counter() - t0
        print(f"{name:<18} n={n:<4} fit={dt:7.2f}s  B={m.boundary_B:<3} cands={len(m.candidates):<4} best={m.best_candidate.get('tag') if m.best_candidate else None}")