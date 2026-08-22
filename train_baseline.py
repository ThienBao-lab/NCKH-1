"""
Baseline 1 pha — huấn luyện bộ phát hiện 9 lớp trên dataset gốc.

Đây là hệ đối chứng của pipeline 2 pha: một model duy nhất vừa tìm người
vừa tìm PPE, rồi gán PPE cho người bằng ĐÚNG thuật toán trong common.py
(việc quy đổi đó do evaluate.py --baseline làm).

Dùng cùng backbone / epoch / imgsz / batch với pha 1 để chênh lệch giữa hai
hệ đến từ kiến trúc chứ không từ dung lượng model.

Chạy:
    python train_baseline.py --data dataset/data.yaml --seed 0
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs_ppe2")
    args = ap.parse_args()

    name = args.name or f"baseline1_seed{args.seed}"
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
        name=name,
        close_mosaic=10,
        **kw,
    )

    print("\n" + "=" * 62)
    print(f"  BASELINE 1 PHA — seed {args.seed} (tập val)")
    print("=" * 62)
    r = m.val(data=args.data, split="val", imgsz=args.imgsz, verbose=False)
    print(f"mAP50 {float(r.box.map50):.4f}   mAP50-95 {float(r.box.map):.4f}   "
          f"P {float(r.box.mp):.4f}   R {float(r.box.mr):.4f}")
    print(f"Checkpoint: {args.project}/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
