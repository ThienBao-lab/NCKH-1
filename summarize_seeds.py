"""
Gộp các file eval_*.json thành bảng mean ± σ trên 3 seed.

Dùng cho Bảng 2 của bài: mỗi ô là trung bình 3 seed kèm độ lệch chuẩn mẫu.
"""

import argparse
import json
import statistics as st
from pathlib import Path

from common import GROUP_NAMES


def load(paths):
    out = []
    for p in paths:
        if Path(p).exists():
            out.append(json.loads(Path(p).read_text()))
    return out


def agg(runs, key):
    """key = 'two_phase' hoặc 'one_phase' -> dict nhóm -> (mean,σ) cho P/R/F1."""
    if not runs or key not in runs[0]:
        return None
    res = {}
    for g in GROUP_NAMES + ["macro"]:
        row = {}
        for m in ("P", "R", "F1"):
            vals = []
            for r in runs:
                if g == "macro":
                    vals.append(r[key]["macro"][m])
                else:
                    vals.append(next(d[m] for d in r[key]["per_group"] if d["group"] == g))
            row[m] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
        res[g] = row
    return res


def show(title, res):
    if res is None:
        print(f"\n[{title}] không có dữ liệu")
        return
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"{'nhóm':<9}{'P':>18}{'R':>18}{'F1':>18}")
    for g in GROUP_NAMES + ["macro"]:
        line = f"{g:<9}"
        for m in ("P", "R", "F1"):
            mu, sd = res[g][m]
            line += f"{mu:>11.4f} ±{sd:<5.4f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="vd 'runs_ppe2/eval_seed{}.json'")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    paths = [args.pattern.format(s) for s in args.seeds.split(",")]
    runs = load(paths)
    print(f"{len(runs)}/{len(paths)} file đọc được: "
          + ", ".join(Path(p).name for p in paths if Path(p).exists()))
    if not runs:
        raise SystemExit("không có file nào")
    print(f"conf pha 1 = {runs[0]['conf1']} | ngưỡng pha 2 = {runs[0]['thresholds']}")

    show(f"{args.title} — 2 PHA (mean ± σ, {len(runs)} seed)", agg(runs, "two_phase"))
    show(f"{args.title} — 1 PHA (mean ± σ, {len(runs)} seed)", agg(runs, "one_phase"))

    t2 = agg(runs, "two_phase")
    t1 = agg(runs, "one_phase")
    if t1:
        d = t2["macro"]["F1"][0] - t1["macro"]["F1"][0]
        print(f"\nChênh macro-F1 (2 pha − 1 pha): {d:+.4f}")


if __name__ == "__main__":
    main()
