"""
Quét ngưỡng conf của pha 1 theo macro-F1 END-TO-END, trên tập VALIDATION.

Vì sao cần: train_phase1.py chọn conf theo P/R của riêng pha 1, với lập luận
"crop thừa sẽ được pha 2 lọc". Lập luận đó không đúng — pha 2 chỉ phân loại
PPE, nó không có lớp "đây không phải người", nên mỗi box thừa mà pha 2 gán
vi phạm sẽ thành FP của cả hệ. Ngưỡng tối ưu cho pha 1 riêng lẻ do đó khác
ngưỡng tối ưu cho toàn hệ, và cái thứ hai mới là cái cần báo cáo.

Quét trên val chứ không phải test — chọn ngưỡng trên tập đánh giá là tự
thổi phồng kết quả.

Chạy:
    python sweep_conf_e2e.py --src dataset --phase1 p1.pt --phase2 best.pt
"""

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

import common
from common import GROUP_NAMES, read_names_from_yaml, configure_classes
from evaluate import IMG_EXT, run_two_phase
from train_phase2 import build_model, build_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-yaml", default="dataset/data.yaml")
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--confs", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--tag", default="phase1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = read_names_from_yaml(args.data_yaml)
    for w in configure_classes(names):
        print(f"⚠ {w}")

    src = Path(args.src)
    img_dir = src / "images" / args.split
    lbl_dir = src / "labels" / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(imgs)} ảnh trong split '{args.split}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.phase2, map_location=device)
    clf = build_model(ck["arch"], pretrained=False)
    clf.load_state_dict(ck["model"])
    clf.to(device).eval()
    tf = build_transforms(False)
    det1 = YOLO(args.phase1)

    thresh = 0.5
    if args.thresholds:
        import json
        d = json.loads(Path(args.thresholds).read_text())["thresholds"]
        thresh = [d[g] for g in GROUP_NAMES]

    confs = [float(c) for c in args.confs.split(",")]
    print(f"\n{'conf':>6}{'P':>10}{'R':>10}{'macroF1':>10}   per-group F1")
    rows, best = [], (-1, None)
    for c in confs:
        t = run_two_phase(imgs, lbl_dir, det1, clf, tf, device, c, thresh)
        Ps, Rs, Fs = [], [], []
        for g in GROUP_NAMES:
            tp, fp, fn = t.tp[g], t.fp[g], t.fn[g]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            Ps.append(p); Rs.append(r); Fs.append(f)
        n = len(GROUP_NAMES)
        mp, mr, mf = sum(Ps)/n, sum(Rs)/n, sum(Fs)/n
        rows.append([c, mp, mr, mf] + Fs)
        if mf > best[0]:
            best = (mf, c)
        print(f"{c:>6.2f}{mp:>10.4f}{mr:>10.4f}{mf:>10.4f}   "
              + " ".join(f"{g}={f:.3f}" for g, f in zip(GROUP_NAMES, Fs)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf", "P", "R", "F1"] + [f"F1_{g}" for g in GROUP_NAMES])
        for r in rows:
            w.writerow([r[0]] + [round(v, 4) for v in r[1:]])
    print(f"\n[{args.tag}] conf tốt nhất trên {args.split}: {best[1]:.2f}  "
          f"(macro-F1 {best[0]:.4f})")
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
