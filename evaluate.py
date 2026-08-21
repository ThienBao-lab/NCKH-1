"""
Đánh giá end-to-end và so sánh công bằng với model 1 pha.

Lớp dương = VI PHẠM. Nhãn GT không xác định (-1) bị bỏ khỏi mọi phép đếm.

Cách tính (áp dụng như nhau cho cả hai hệ):
  TP  người khớp GT, GT=vi phạm, dự đoán=vi phạm
  FP  người khớp GT, GT=tuân thủ, dự đoán=vi phạm
      + người phát hiện thừa (không khớp GT nào) mà dự đoán=vi phạm
  FN  người khớp GT, GT=vi phạm, dự đoán=tuân thủ
      + người GT bị bỏ sót hoàn toàn mà GT=vi phạm      <-- lỗi pha 1

Chạy:
    python evaluate.py --src DATASET --phase1 p1.pt --phase2 best.pt \
                       --baseline model_1pha.pt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

import common
from common import (
    CROP_MARGIN, GROUP_NAMES, MIN_CROP_SIDE, N_GROUPS,
    assign_ppe_to_persons, configure_classes, expand_box, iou,
    read_names_from_yaml, read_yolo_label, tally_to_labels,
)
from train_phase2 import build_model, build_transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MATCH_IOU = 0.5


# ---------------------------------------------------------------- đếm
class Tally:
    def __init__(self):
        self.tp = Counter(); self.fp = Counter(); self.fn = Counter()

    def add(self, g, gt, pred):
        if gt < 0:
            return
        if pred == 1 and gt == 1: self.tp[g] += 1
        elif pred == 1 and gt == 0: self.fp[g] += 1
        elif pred == 0 and gt == 1: self.fn[g] += 1

    def report(self, title):
        print("\n" + "=" * 52)
        print(f"  {title}")
        print("=" * 52)
        print(f"{'nhóm':<10}{'P':>10}{'R':>10}{'F1':>10}{'#GT+':>8}")
        Ps, Rs, Fs, data = [], [], [], []
        for g in GROUP_NAMES:
            tp, fp, fn = self.tp[g], self.fp[g], self.fn[g]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            Ps.append(p); Rs.append(r); Fs.append(f)
            data.append({"group": g, "P": round(p, 4), "R": round(r, 4),
                         "F1": round(f, 4), "TP": tp, "FP": fp, "FN": fn})
            print(f"{g:<10}{p:>10.4f}{r:>10.4f}{f:>10.4f}{tp+fn:>8}")
        n = len(GROUP_NAMES)
        macro = {"P": round(sum(Ps)/n, 4), "R": round(sum(Rs)/n, 4),
                 "F1": round(sum(Fs)/n, 4)}
        print(f"{'macro':<10}{macro['P']:>10.4f}{macro['R']:>10.4f}{macro['F1']:>10.4f}")
        return {"title": title, "per_group": data, "macro": macro}


def match_persons(pred_boxes, gt_boxes):
    """Ghép tham lam theo IoU. -> (dict pred_i->gt_j, set gt chưa khớp)."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(pred_boxes) for j, g in enumerate(gt_boxes)),
        reverse=True,
    )
    m, used_p, used_g = {}, set(), set()
    for s, i, j in pairs:
        if s < MATCH_IOU or i in used_p or j in used_g:
            continue
        m[i] = j; used_p.add(i); used_g.add(j)
    return m, set(range(len(gt_boxes))) - used_g


# ---------------------------------------------------------------- pha 2
@torch.no_grad()
def classify(model, tf, image, boxes, device, thresh=0.5):
    """Trả list nhãn dự đoán (0/1) cho từng box người."""
    W, H = image.size
    out = []
    batch, keep = [], []
    for k, b in enumerate(boxes):
        ex = expand_box(b, CROP_MARGIN, W, H)
        if min(ex[2] - ex[0], ex[3] - ex[1]) < MIN_CROP_SIDE:
            continue
        batch.append(tf(image.crop((int(ex[0]), int(ex[1]), int(ex[2]), int(ex[3])))))
        keep.append(k)

    preds = [[0] * N_GROUPS for _ in boxes]      # crop quá nhỏ -> mặc định tuân thủ
    if batch:
        x = torch.stack(batch).to(device)
        p = (torch.sigmoid(model(x)).cpu() > thresh).int().tolist()
        for k, row in zip(keep, p):
            preds[k] = row
    return preds


# ---------------------------------------------------------------- chạy
def run_two_phase(imgs, lbl_dir, det1, clf, tf, device, conf):
    t = Tally()
    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        gt_persons, gt_ppes = read_yolo_label(lbl_dir / (ip.stem + ".txt"), W, H)
        if not gt_persons:
            continue
        gt_labels = [tally_to_labels(x)
                     for x in assign_ppe_to_persons(gt_persons, gt_ppes)]

        r = det1.predict(str(ip), conf=conf, verbose=False)[0]
        pred_boxes = [tuple(b) for b in r.boxes.xyxy.cpu().numpy().tolist()]
        pred_labels = classify(clf, tf, im, pred_boxes, device) if pred_boxes else []

        m, missed = match_persons(pred_boxes, gt_persons)
        for i, j in m.items():
            for k, g in enumerate(GROUP_NAMES):
                t.add(g, gt_labels[j][k], pred_labels[i][k])
        for i in range(len(pred_boxes)):          # người phát hiện thừa
            if i not in m:
                for k, g in enumerate(GROUP_NAMES):
                    if pred_labels[i][k] == 1:
                        t.fp[g] += 1
        for j in missed:                          # người bị pha 1 bỏ sót
            for k, g in enumerate(GROUP_NAMES):
                if gt_labels[j][k] == 1:
                    t.fn[g] += 1
    return t


def run_one_phase(imgs, lbl_dir, det9, conf):
    """Model 1 pha -> quy về nhãn theo từng người bằng ĐÚNG thuật toán gán."""
    t = Tally()
    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        gt_persons, gt_ppes = read_yolo_label(lbl_dir / (ip.stem + ".txt"), W, H)
        if not gt_persons:
            continue
        gt_labels = [tally_to_labels(x)
                     for x in assign_ppe_to_persons(gt_persons, gt_ppes)]

        r = det9.predict(str(ip), conf=conf, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        pred_boxes = [tuple(b) for c, b in zip(cls, xyxy) if c == common.PERSON_ID]
        pred_ppes = [(c, tuple(b)) for c, b in zip(cls, xyxy) if c != common.PERSON_ID]
        pred_labels = [tally_to_labels(x)
                       for x in assign_ppe_to_persons(pred_boxes, pred_ppes)]
        # model 1 pha không nói "không xác định" -> coi như tuân thủ
        pred_labels = [[0 if v < 0 else v for v in row] for row in pred_labels]

        m, missed = match_persons(pred_boxes, gt_persons)
        for i, j in m.items():
            for k, g in enumerate(GROUP_NAMES):
                t.add(g, gt_labels[j][k], pred_labels[i][k])
        for i in range(len(pred_boxes)):
            if i not in m:
                for k, g in enumerate(GROUP_NAMES):
                    if pred_labels[i][k] == 1:
                        t.fp[g] += 1
        for j in missed:
            for k, g in enumerate(GROUP_NAMES):
                if gt_labels[j][k] == 1:
                    t.fn[g] += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="dataset gốc theo bố cục images/labels/{split}")
    ap.add_argument("--test-dir", help="thư mục chứa images/ và labels/ của test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--phase1", required=True, help="best.pt của pha 1")
    ap.add_argument("--phase2", required=True, help="best.pt của pha 2")
    ap.add_argument("--baseline", help="best.pt model 1 pha 9 lớp (tuỳ chọn)")
    ap.add_argument("--conf1", type=float, default=0.20)
    ap.add_argument("--conf-baseline", type=float, default=0.25)
    ap.add_argument("--data-yaml", help="data.yaml gốc, để lấy đúng thứ tự lớp")
    ap.add_argument("--save-json", help="lưu kết quả ra JSON để vẽ hình")
    args = ap.parse_args()

    if args.data_yaml:
        names = read_names_from_yaml(args.data_yaml)
        print("Thứ tự lớp:", names)
        for w in configure_classes(names):
            print(f"⚠ {w}")
    else:
        print("⚠ Không có --data-yaml, dùng thứ tự lớp mặc định trong common.py")

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.test_dir:
        base = Path(args.test_dir)
        img_dir, lbl_dir = base / "images", base / "labels"
    elif args.src:
        src = Path(args.src)
        img_dir = src / "images" / args.split
        lbl_dir = src / "labels" / args.split
    else:
        raise SystemExit("cần --src hoặc --test-dir")
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(imgs)} ảnh trong split '{args.split}'")

    ck = torch.load(args.phase2, map_location=device)
    clf = build_model(ck["arch"], pretrained=False)
    clf.load_state_dict(ck["model"])
    clf.to(device).eval()
    tf = build_transforms(False)

    t2 = run_two_phase(imgs, lbl_dir, YOLO(args.phase1), clf, tf, device, args.conf1)
    r2 = t2.report(f"2 PHA — end-to-end (conf pha 1 = {args.conf1})")

    out = {"two_phase": r2, "conf1": args.conf1,
           "phase1": args.phase1, "phase2": args.phase2}

    if args.baseline:
        t1 = run_one_phase(imgs, lbl_dir, YOLO(args.baseline), args.conf_baseline)
        r1 = t1.report("1 PHA — quy về nhãn theo từng người")
        out["one_phase"] = r1
        out["baseline"] = args.baseline
        f2, f1 = r2["macro"]["F1"], r1["macro"]["F1"]
        print("\n" + "=" * 52)
        print(f"  macro-F1:  2 pha {f2:.4f}   |   1 pha {f1:.4f}   "
              f"|   chênh {f2-f1:+.4f}")
        print("=" * 52)

    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nĐã lưu: {p}")


if __name__ == "__main__":
    main()