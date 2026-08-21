"""
Xuất ảnh minh hoạ định tính cho bài báo.

Với mỗi ảnh test được chọn, vẽ:
  - khung người pha 1 phát hiện được
  - nhãn vi phạm pha 2 dự đoán cho từng người
  - đối chiếu với nhãn thật (đúng màu xanh, sai màu đỏ)

Ưu tiên chọn những ảnh "kể được câu chuyện": ảnh mà hệ 2 pha đúng còn
nhãn thật có vi phạm ở nhóm vật nhỏ (gloves/boots) — đó là chỗ phương
pháp tạo khác biệt.

Chạy:
    python export_examples.py --src dataset --phase1 p1.pt --phase2 p2.pt \
        --out figures/examples --n 12
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import common
from common import (
    CROP_MARGIN, GROUP_NAMES, MIN_CROP_SIDE, assign_ppe_to_persons,
    configure_classes, expand_box, iou, read_names_from_yaml, read_yolo_label,
    tally_to_labels,
)
from evaluate import classify, match_persons
from train_phase2 import build_model, build_transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OK, BAD, NEU = (34, 197, 94), (239, 68, 68), (148, 163, 184)
SHORT = {"helmet": "mũ", "vest": "áo", "gloves": "găng", "boots": "giày"}


def draw_one(im, persons, preds, gts, scale=1.0):
    """Vẽ khung + nhãn lên bản sao của ảnh."""
    im = im.copy()
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    d = ImageDraw.Draw(im)

    for k, box in enumerate(persons):
        b = [v * scale for v in box]
        pred, gt = preds[k], gts[k]

        # màu khung: xanh nếu mọi nhãn xác định đều đúng
        checked = [(p, g) for p, g in zip(pred, gt) if g >= 0]
        if not checked:
            colour = NEU
        elif all(p == g for p, g in checked):
            colour = OK
        else:
            colour = BAD
        d.rectangle(b, outline=colour, width=3)

        # nhãn: chỉ ghi nhóm có vi phạm hoặc dự đoán sai
        lines = []
        for i, gname in enumerate(GROUP_NAMES):
            p, g = pred[i], gt[i]
            if g < 0:
                continue
            if g == 1 or p != g:
                mark = "✓" if p == g else "✗"
                lines.append(f"{mark}{SHORT[gname]}")
        if lines:
            txt = " ".join(lines)
            ty = max(0, b[1] - 16)
            d.rectangle([b[0], ty, b[0] + 7 * len(txt) + 6, ty + 15], fill=colour)
            d.text((b[0] + 3, ty + 2), txt, fill="white")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="dataset gốc theo bố cục images/labels/{split}")
    ap.add_argument("--test-dir", help="thư mục chứa images/ và labels/ của test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-yaml", help="data.yaml gốc")
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--conf1", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=12, help="số ảnh xuất ra")
    ap.add_argument("--min-persons", type=int, default=2,
                    help="chỉ chọn ảnh có ít nhất N người")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="figures/examples")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.data_yaml:
        for w in configure_classes(read_names_from_yaml(args.data_yaml)):
            print(f"⚠ {w}")

    if args.test_dir:
        base = Path(args.test_dir)
        img_dir, lbl_dir = base / "images", base / "labels"
    elif args.src:
        img_dir = Path(args.src) / "images" / args.split
        lbl_dir = Path(args.src) / "labels" / args.split
    else:
        raise SystemExit("cần --src hoặc --test-dir")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.phase2, map_location=device)
    clf = build_model(ck["arch"], pretrained=False)
    clf.load_state_dict(ck["model"])
    clf.to(device).eval()
    tf = build_transforms(False)
    det = YOLO(args.phase1)

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(imgs)} ảnh, đang chọn {args.n} ảnh minh hoạ...")

    cands = []
    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        gt_persons, gt_ppes = read_yolo_label(lbl_dir / (ip.stem + ".txt"), W, H)
        if len(gt_persons) < args.min_persons:
            continue
        gt_labels = [tally_to_labels(t)
                     for t in assign_ppe_to_persons(gt_persons, gt_ppes)]

        r = det.predict(str(ip), conf=args.conf1, verbose=False)[0]
        pred_boxes = [tuple(b) for b in r.boxes.xyxy.cpu().numpy().tolist()]
        if not pred_boxes:
            continue
        preds = classify(clf, tf, im, pred_boxes, device)

        m, _ = match_persons(pred_boxes, gt_persons)
        if not m:
            continue

        # điểm ưu tiên: nhiều nhãn đúng + có vi phạm ở nhóm vật nhỏ
        n_ok = n_all = n_small = 0
        for i, j in m.items():
            for k in range(len(GROUP_NAMES)):
                g = gt_labels[j][k]
                if g < 0:
                    continue
                n_all += 1
                n_ok += int(preds[i][k] == g)
                if g == 1 and GROUP_NAMES[k] in ("gloves", "boots"):
                    n_small += 1
        if n_all == 0:
            continue
        score = n_ok / n_all + 0.3 * min(n_small, 3)

        boxes = [pred_boxes[i] for i in m]
        pp = [preds[i] for i in m]
        gg = [gt_labels[j] for j in m.values()]
        cands.append((score, ip, im, boxes, pp, gg))

    cands.sort(key=lambda t: -t[0])
    for rank, (score, ip, im, boxes, pp, gg) in enumerate(cands[:args.n], 1):
        vis = draw_one(im, boxes, pp, gg, args.scale)
        name = f"vd{rank:02d}_{ip.stem}.jpg"
        vis.save(out / name, quality=94)
        print(f"  {name}   ({len(boxes)} người, điểm {score:.2f})")

    print(f"\nĐã xuất {min(len(cands), args.n)} ảnh vào {out}/")
    print("Chú thích: khung xanh = mọi nhãn đúng, đỏ = có nhãn sai,"
          " xám = không có nhãn xác định.")
    print("Chữ trên khung: ✓/✗ + tên nhóm, chỉ hiện nhóm có vi phạm hoặc dự đoán sai.")


if __name__ == "__main__":
    main()
