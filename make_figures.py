"""
Vẽ toàn bộ hình cho bài báo từ các file JSON/CSV đã sinh ra.

Đầu vào (chỉ cần cái nào có, thiếu thì bỏ qua hình tương ứng):
  - eval_seed*.json      từ evaluate.py --save-json
  - thresholds.json      từ tune_thresholds.py
  - conf_sweep.csv       từ train_phase1.py
  - stats.json           từ prepare_data.py

Chạy:
    python make_figures.py --results runs_ppe2 --out figures
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import GROUP_NAMES

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

C_TWO, C_ONE = "#2563eb", "#dc2626"


def save(fig, out, name):
    for ext in ("png", "pdf"):          # pdf để nhúng vào LaTeX
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}.png / .pdf")


# ---------------------------------------------------------------- hình 1
def fig_per_group(runs, out):
    """So sánh P/R/F1 từng nhóm: 2 pha vs 1 pha, có thanh sai số nếu n>1."""
    if not runs:
        return
    has_one = all("one_phase" in r for r in runs)

    def gather(key, metric):
        # (n_run, n_group)
        return np.array([[g[metric] for g in r[key]["per_group"]] for r in runs])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(GROUP_NAMES))
    w = 0.36

    for ax, metric in zip(axes, ("P", "R", "F1")):
        two = gather("two_phase", metric)
        ax.bar(x - w/2 if has_one else x, two.mean(0), w if has_one else w*1.6,
               yerr=two.std(0) if len(runs) > 1 else None, capsize=3,
               label="2 pha", color=C_TWO)
        if has_one:
            one = gather("one_phase", metric)
            ax.bar(x + w/2, one.mean(0), w,
                   yerr=one.std(0) if len(runs) > 1 else None, capsize=3,
                   label="1 pha", color=C_ONE)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES, rotation=20, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Điểm")
    axes[0].legend(loc="upper right", framealpha=0.9)
    n = len(runs)
    fig.suptitle(f"So sánh theo từng nhóm PPE trên tập test"
                 + (f" (trung bình {n} seed)" if n > 1 else ""), y=1.02)
    save(fig, out, "fig_per_group")


# ---------------------------------------------------------------- hình 2
def fig_macro(runs, out):
    """Macro P/R/F1 tổng: 2 pha vs 1 pha."""
    if not runs or not all("one_phase" in r for r in runs):
        return
    metrics = ("P", "R", "F1")
    two = np.array([[r["two_phase"]["macro"][m] for m in metrics] for r in runs])
    one = np.array([[r["one_phase"]["macro"][m] for m in metrics] for r in runs])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x, w = np.arange(3), 0.35
    b1 = ax.bar(x - w/2, two.mean(0), w, yerr=two.std(0) if len(runs) > 1 else None,
                capsize=4, label="2 pha", color=C_TWO)
    b2 = ax.bar(x + w/2, one.mean(0), w, yerr=one.std(0) if len(runs) > 1 else None,
                capsize=4, label="1 pha", color=C_ONE)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro trung bình")
    ax.legend()
    ax.set_title("Kết quả tổng trên tập test")
    save(fig, out, "fig_macro")


# ---------------------------------------------------------------- hình 3
def fig_conf_sweep(path, out):
    """Đường cong P/R/F1 theo ngưỡng conf của pha 1."""
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return
    conf = [float(r["conf"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, color, mark in (("P", "#7c3aed", "o"), ("R", "#059669", "s"),
                             ("F1", "#ea580c", "^")):
        ax.plot(conf, [float(r[key]) for r in rows], marker=mark,
                color=color, label=key, linewidth=1.8, markersize=5)
    ax.set_xlabel("Ngưỡng conf của pha 1")
    ax.set_ylabel("Điểm")
    ax.set_title("Ảnh hưởng của ngưỡng conf lên phát hiện người")
    ax.legend()
    save(fig, out, "fig_conf_sweep")


# ---------------------------------------------------------------- hình 4
def fig_thresholds(path, out):
    """F1 trước/sau khi tối ưu ngưỡng riêng từng nhóm."""
    if not path.exists():
        return
    d = json.loads(path.read_text())
    rows = d.get("per_group_test", [])
    if not rows:
        return
    names = [r["group"] for r in rows]
    old = [r["F1_default"] for r in rows]
    new = [r["F1_tuned"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x, w = np.arange(len(names)), 0.35
    b1 = ax.bar(x - w/2, old, w, label="ngưỡng chung 0,5", color="#94a3b8")
    b2 = ax.bar(x + w/2, new, w, label="ngưỡng riêng", color=C_TWO)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    for i, r in enumerate(rows):
        ax.text(i + w/2, 0.03, f"t={r['threshold']:.2f}",
                ha="center", fontsize=8, color="white", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 trên tập test")
    ax.legend()
    ax.set_title("Tối ưu ngưỡng quyết định riêng cho từng nhóm")
    save(fig, out, "fig_thresholds")


# ---------------------------------------------------------------- hình 5
def fig_label_stats(path, out):
    """Phân bố nhãn: tuân thủ / vi phạm / không xác định."""
    if not path.exists():
        return
    s = json.loads(path.read_text())
    ok = [s.get(f"label_{g}_ok", 0) for g in GROUP_NAMES]
    vi = [s.get(f"label_{g}_violation", 0) for g in GROUP_NAMES]
    un = [s.get(f"label_{g}_unknown", 0) for g in GROUP_NAMES]
    if sum(ok) + sum(vi) + sum(un) == 0:
        return

    tot = np.array(ok) + np.array(vi) + np.array(un)
    tot[tot == 0] = 1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = np.zeros(len(GROUP_NAMES))
    for vals, lab, col in ((ok, "tuân thủ", "#059669"),
                           (vi, "vi phạm", "#dc2626"),
                           (un, "không xác định", "#cbd5e1")):
        pct = np.array(vals) / tot * 100
        ax.bar(GROUP_NAMES, pct, bottom=b, label=lab, color=col)
        b += pct
    ax.set_ylabel("Tỉ lệ (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Phân bố nhãn theo nhóm PPE")
    save(fig, out, "fig_label_stats")


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs_ppe2",
                    help="thư mục chứa eval_seed*.json, thresholds.json...")
    ap.add_argument("--stats", default=None,
                    help="đường dẫn stats.json của prepare_data (mặc định tự dò)")
    ap.add_argument("--conf-sweep", default=None,
                    help="đường dẫn conf_sweep.csv (mặc định tự dò)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for f in sorted(res.glob("eval_seed*.json")) or sorted(res.glob("eval*.json")):
        runs.append(json.loads(f.read_text()))
    print(f"Đọc được {len(runs)} file kết quả đánh giá")

    sweep = Path(args.conf_sweep) if args.conf_sweep else next(
        iter(sorted(res.glob("**/conf_sweep.csv"))), res / "conf_sweep.csv")
    stats = Path(args.stats) if args.stats else next(
        iter(sorted(res.parent.glob("**/stats.json"))), res / "stats.json")

    print("\nĐang vẽ:")
    fig_per_group(runs, out)
    fig_macro(runs, out)
    fig_conf_sweep(sweep, out)
    fig_thresholds(res / "thresholds.json", out)
    fig_label_stats(stats, out)

    made = sorted(p.name for p in out.glob("*.png"))
    if made:
        print(f"\nĐã tạo {len(made)} hình trong {out}/")
        print("File .pdf đi kèm dùng để nhúng vào LaTeX (nét hơn .png).")
    else:
        print("\nKhông vẽ được hình nào — kiểm tra lại đường dẫn --results.")


if __name__ == "__main__":
    main()
