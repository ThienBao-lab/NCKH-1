"""
Pha 1 — huấn luyện bộ phát hiện người 1 lớp.

Chạy:
    python train_phase1.py --data OUT/phase1/data.yaml
Sau khi train sẽ quét ngưỡng conf để bạn chọn điểm vận hành:
người bị bỏ sót ở pha 1 là mất vĩnh viễn, nên ưu tiên Recall hơn Precision.
"""

import argparse

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
    print(f"{'conf':>6}{'P':>10}{'R':>10}{'mAP50':>10}")
    for conf in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        r = m.val(data=args.data, split="test", imgsz=args.imgsz,
                  conf=conf, verbose=False)
        print(f"{conf:>6.2f}{r.box.mp:>10.4f}{r.box.mr:>10.4f}{r.box.map50:>10.4f}")

    print("\nChọn conf có Recall cao nhất mà Precision còn chấp nhận được —")
    print("crop thừa sẽ được pha 2 lọc, còn người bị bỏ sót thì không cứu được.")


if __name__ == "__main__":
    main()
