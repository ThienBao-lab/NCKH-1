"""
Hình: ngưỡng conf tối ưu cho pha 1 RIÊNG LẺ khác ngưỡng tối ưu cho TOÀN HỆ.

Trục trái: Recall của pha 1 (giảm đơn điệu theo conf).
Trục phải: macro-F1 end-to-end của cả hệ (tăng rồi mới giảm).
Hai đường đạt cực trị ở hai chỗ khác nhau — đó là lý do không được chọn
ngưỡng pha 1 bằng chỉ tiêu của riêng pha 1.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures")


def read(path, cols):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return {c: [float(r[c]) for r in rows] for c in cols} | {
        "conf": [float(r["conf"]) for r in rows]}


p1 = read("runs_ppe2/phase1_person/conf_sweep_val.csv", ["R"])
e2e = read("runs_ppe2/phase1_person/conf_sweep_e2e_val.csv", ["F1", "P", "R"])
if p1 is None or e2e is None:
    raise SystemExit("thiếu file csv")

fig, ax = plt.subplots(figsize=(7, 4.4))
l1, = ax.plot(p1["conf"], p1["R"], "o-", color="#d62728",
              label="Recall của riêng pha 1 (val)")
ax.set_xlabel("ngưỡng conf của pha 1")
ax.set_ylabel("Recall pha 1", color="#d62728")
ax.tick_params(axis="y", labelcolor="#d62728")
ax.set_ylim(0.5, 1.0)
ax.grid(alpha=0.3)

ax2 = ax.twinx()
l2, = ax2.plot(e2e["conf"], e2e["F1"], "s-", color="#1f77b4",
               label="macro-F1 end-to-end (val)")
ax2.set_ylabel("macro-F1 toàn hệ", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_ylim(0.5, 1.0)

best_i = max(range(len(e2e["conf"])), key=lambda i: e2e["F1"][i])
best_c = e2e["conf"][best_i]
ax2.axvline(best_c, ls="--", color="#1f77b4", alpha=0.6)
ax2.annotate(f"tối ưu toàn hệ\nconf = {best_c:.2f}",
             (best_c, e2e["F1"][best_i]), textcoords="offset points",
             xytext=(-108, 20), color="#1f77b4", fontsize=9)

best_r = max(range(len(p1["conf"])), key=lambda i: p1["R"][i])
ax.axvline(p1["conf"][best_r], ls=":", color="#d62728", alpha=0.6)
ax.annotate(f"tối ưu recall pha 1\nconf = {p1['conf'][best_r]:.2f}",
            (p1["conf"][best_r], p1["R"][best_r]), textcoords="offset points",
            xytext=(-30, -62), color="#d62728", fontsize=9)

ax.set_title("Ngưỡng tối ưu cho pha 1 riêng lẻ ≠ ngưỡng tối ưu cho toàn hệ")
ax.legend(handles=[l1, l2], loc="lower left", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_conf_e2e.{ext}", dpi=150)
print("✓ fig_conf_e2e.png / .pdf")
