"""
Pha 1 — huấn luyện bộ phát hiện người 1 lớp.

Chạy:
    python train_phase1.py --data OUT/phase1/data.yaml
Sau khi train sẽ quét ngưỡng conf để bạn chọn điểm vận hành:
người bị bỏ sót ở pha 1 là mất vĩnh viễn, nên ưu tiên Recall hơn Precision.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="'0' hoặc '1' cho 1 GPU; '0,1' để dùng DDP cả hai")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"],
                    help="cache ảnh vào RAM — máy nhiều RAM nên bật")
    ap.add_argument("--name", default="phase1_person")
    ap.add_argument("--project", default="runs_ppe2")
    args = ap.parse_args()

    kw = {}
    if args.device is not None:
        kw["device"] = [int(d) for d in args.device.split(",")] \
            if "," in args.device else int(args.device)

    m = YOLO(args.model)
    m.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
        amp=True,
        project=args.project,
        name=args.name,
        close_mosaic=10,
        **kw,
    )

    print("\n" + "=" * 58)
    print("  QUÉT NGƯỠNG CONF (tập test)")
    print("=" * 58)
    print(f"{'conf':>6}{'P':>10}{'R':>10}{'F1':>10}{'mAP50':>10}")

    rows, best_f1, best_conf = [], -1.0, None
    for i in range(10):                       # 0.10 -> 0.55, bước 0.05
        conf = round(0.10 + 0.05 * i, 2)
        r = m.val(data=args.data, split="test", imgsz=args.imgsz,
                  conf=conf, verbose=False)
        p, rec = float(r.box.mp), float(r.box.mr)
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        rows.append((conf, p, rec, f1, float(r.box.map50)))
        if f1 > best_f1:
            best_f1, best_conf = f1, conf

    for conf, p, rec, f1, m50 in rows:
        mark = "  <- F1 cao nhat" if conf == best_conf else ""
        print(f"{conf:>6.2f}{p:>10.4f}{rec:>10.4f}{f1:>10.4f}{m50:>10.4f}{mark}")

    csv_path = Path(args.project) / args.name / "conf_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf", "P", "R", "F1", "mAP50"])
        for row in rows:
            w.writerow([row[0]] + [round(v, 4) for v in row[1:]])
    print(f"\nĐã lưu bảng: {csv_path}")

    print("\nChọn conf có Recall cao nhất mà Precision còn chấp nhận được —")
    print("crop thừa sẽ được pha 2 lọc, còn người bị bỏ sót thì không cứu được.")


if __name__ == "__main__":
    main()