"""
Xuất ảnh minh hoạ định tính cho bài báo.

Với mỗi ảnh test được chọn, vẽ:
  - khung người pha 1 phát hiện được
  - nhãn vi phạm pha 2 dự đoán cho từng người, ghi rõ thiếu / sót / nhầm
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
from PIL import Image, ImageDraw, ImageFont

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
MISS = (245, 158, 11)          # người có trong nhãn mà pha 1 không tìm ra
SHORT = {"helmet": "mũ", "vest": "áo", "gloves": "găng", "boots": "giày"}

# Font bitmap mặc định của PIL không có glyph tiếng Việt lẫn ✓/✗ (chữ ra thành
# ô vuông) và không phóng to theo --scale. DejaVu có đủ cả hai.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def dashed_rect(d, box, colour, width, dash=9):
    """Khung đứt nét — dùng cho người pha 1 bỏ sót."""
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=colour, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=colour, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=colour, width=width)


def draw_one(im, persons, preds, gts, scale=1.0, missed_boxes=()):
    """Vẽ khung + nhãn lên bản sao của ảnh."""
    im = im.copy()
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    d = ImageDraw.Draw(im)
    # cỡ chữ theo cạnh ảnh để ảnh lớn nhỏ đều đọc được
    fs = max(13, int(min(im.width, im.height) * 0.022))
    font = load_font(fs)
    pad, lw = max(2, fs // 4), max(2, int(3 * scale))
    placed = []                                   # vùng nhãn đã vẽ, để tránh chồng

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
        d.rectangle(b, outline=colour, width=lw)

        # Nhãn: chỉ ghi nhóm có vi phạm hoặc dự đoán sai.
        # Phải nói rõ NỘI DUNG chứ không chỉ đúng/sai: "✓mũ" dễ bị đọc thành
        # "có đội mũ", trong khi ý là "thiếu mũ và model bắt đúng".
        lines = []
        for i, gname in enumerate(GROUP_NAMES):
            p, g = pred[i], gt[i]
            if g < 0:
                continue
            if g == 1 and p == 1:                 # vi phạm, bắt đúng
                lines.append(f"✓thiếu {SHORT[gname]}")
            elif g == 1 and p == 0:               # vi phạm, bỏ sót
                lines.append(f"✗sót {SHORT[gname]}")
            elif g == 0 and p == 1:               # tuân thủ, báo nhầm
                lines.append(f"✗nhầm {SHORT[gname]}")
        if lines:
            # mỗi nhóm một dòng: nhãn hẹp hơn nhiều so với ghép một dòng dài,
            # nên dễ xếp mà không đè lên người bên cạnh
            txt = "\n".join(lines)
            x0, y0, x1, y1 = d.multiline_textbbox((0, 0), txt, font=font,
                                                  spacing=2)
            tw, th = x1 - x0, y1 - y0
            bx = max(0, min(b[0], im.width - tw - 2 * pad))   # không tràn hai bên
            by = b[1] - th - 2 * pad
            if by < 0:                                        # hết chỗ phía trên
                by = b[1]
            box_t = [bx, by, bx + tw + 2 * pad, by + th + 2 * pad]

            # đẩy xuống cho tới khi không đè nhãn nào đã vẽ
            for _ in range(len(placed) + 1):
                hit = next((pr for pr in placed
                            if not (box_t[2] <= pr[0] or box_t[0] >= pr[2]
                                    or box_t[3] <= pr[1] or box_t[1] >= pr[3])),
                           None)
                if hit is None:
                    break
                shift = hit[3] - box_t[1] + 2
                box_t[1] += shift
                box_t[3] += shift

            d.rectangle(box_t, fill=colour)
            d.multiline_text((box_t[0] + pad, box_t[1] + pad - y0), txt,
                             fill="white", font=font, spacing=2)
            placed.append(box_t)

    # Người có trong nhãn mà pha 1 không tìm ra. Không vẽ thì hình che mất
    # đúng cái nút thắt của kiến trúc 2 pha — mọi vi phạm của họ đều thành FN.
    for box in missed_boxes:
        b = [v * scale for v in box]
        dashed_rect(d, b, MISS, lw)
        txt = "pha 1 bỏ sót"
        x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=font)
        tw, th = x1 - x0, y1 - y0
        bx = max(0, min(b[0], im.width - tw - 2 * pad))
        by = b[1] - th - 2 * pad
        if by < 0:
            by = b[1]
        box_t = [bx, by, bx + tw + 2 * pad, by + th + 2 * pad]
        for _ in range(len(placed) + 1):
            hit = next((pr for pr in placed
                        if not (box_t[2] <= pr[0] or box_t[0] >= pr[2]
                                or box_t[3] <= pr[1] or box_t[1] >= pr[3])), None)
            if hit is None:
                break
            shift = hit[3] - box_t[1] + 2
            box_t[1] += shift
            box_t[3] += shift
        d.rectangle(box_t, fill=MISS)
        d.text((box_t[0] + pad, box_t[1] + pad - y0), txt, fill="white", font=font)
        placed.append(box_t)
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
    ap.add_argument("--boxes", default="phase1", choices=["phase1", "gt"],
                    help="'phase1' = box do pha 1 tìm (end-to-end, có vẽ cả "
                         "người bị bỏ sót); 'gt' = box người lấy từ nhãn gốc "
                         "(protocol oracle, chỉ đánh giá pha 2)")
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

        if args.boxes == "gt":
            # Protocol oracle: box người lấy thẳng từ nhãn, pha 1 không tham gia.
            # Hình khi đó cho thấy đúng năng lực nhận PPE của pha 2, không bị
            # số người pha 1 bỏ sót làm nhiễu.
            pred_boxes = list(gt_persons)
            m = {i: i for i in range(len(gt_persons))}
            missed = set()
        else:
            r = det.predict(str(ip), conf=args.conf1, verbose=False)[0]
            pred_boxes = [tuple(b) for b in r.boxes.xyxy.cpu().numpy().tolist()]
            if not pred_boxes:
                continue
            m, missed = match_persons(pred_boxes, gt_persons)
            if not m:
                continue
        preds = classify(clf, tf, im, pred_boxes, device)

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
        miss_boxes = [gt_persons[j] for j in sorted(missed)]
        cands.append((score, ip, im, boxes, pp, gg, miss_boxes))

    cands.sort(key=lambda t: -t[0])
    for rank, (score, ip, im, boxes, pp, gg, mb) in enumerate(cands[:args.n], 1):
        vis = draw_one(im, boxes, pp, gg, args.scale, mb)
        name = f"vd{rank:02d}_{ip.stem}.jpg"
        vis.save(out / name, quality=94)
        note = f", {len(mb)} người pha 1 bỏ sót" if mb else ""
        print(f"  {name}   ({len(boxes)} người{note}, điểm {score:.2f})")

    print(f"\nĐã xuất {min(len(cands), args.n)} ảnh vào {out}/")
    print("\nChú thích để dùng dưới hình trong bài:")
    print("  Khung xanh = mọi nhãn xác định đều đúng · đỏ = có nhãn sai ·"
          " xám = không nhãn nào xác định được.")
    print("  ✓thiếu X = người này KHÔNG mang X, model bắt đúng.")
    print("  ✗sót X   = người này không mang X nhưng model không báo (FN).")
    print("  ✗nhầm X  = người này CÓ mang X nhưng model báo vi phạm (FP).")
    print("  Nhóm không xác định được từ nhãn gốc thì không hiện.")


if __name__ == "__main__":
    main()
