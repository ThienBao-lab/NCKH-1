"""
Sinh dữ liệu cho pipeline 2 pha từ dataset PPE gốc (9 lớp, định dạng YOLO).

Tạo ra 2 thứ:
  1) dataset phát hiện người 1 lớp  -> huấn luyện pha 1
  2) dataset crop + nhãn nhị phân   -> huấn luyện pha 2

Chạy:
    python prepare_data.py --src /path/dataset --out /path/out
Cấu trúc `src` mong đợi (chuẩn Ultralytics):
    src/images/{train,val,test}/*.jpg
    src/labels/{train,val,test}/*.txt
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import common
from common import (
    CROP_MARGIN, GROUP_NAMES, MIN_CROP_SIDE, OVERLAP_IOU,
    assign_ppe_to_persons, configure_classes, expand_box, iou,
    read_names_from_yaml, read_yolo_label, tally_to_labels,
)

SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_label(labels_dir, img_path):
    return labels_dir / (img_path.stem + ".txt")


def build_phase1(out, split, img_paths, labels_dir):
    """Dataset 1 lớp: giữ mỗi box person, remap class id về 0."""
    img_out = out / "phase1" / "images" / split
    lbl_out = out / "phase1" / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for ip in img_paths:
        lp = find_label(labels_dir, ip)
        if not lp.exists():
            continue
        keep = []
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) == common.PERSON_ID:
                keep.append("0 " + " ".join(parts[1:5]))
        if not keep:
            continue
        # symlink ảnh cho nhẹ đĩa; đổi sang copy nếu hệ thống không hỗ trợ
        dst = img_out / ip.name
        if not dst.exists():
            try:
                dst.symlink_to(ip.resolve())
            except OSError:
                dst.write_bytes(ip.read_bytes())
        (lbl_out / (ip.stem + ".txt")).write_text("\n".join(keep) + "\n")
        n_img += 1
        n_box += len(keep)
    return n_img, n_box


def build_phase2(out, split, img_paths, labels_dir, args, writer, stats):
    """Cắt crop từng người + suy ra 4 nhãn nhị phân."""
    crop_dir = out / "phase2" / "crops" / split
    crop_dir.mkdir(parents=True, exist_ok=True)

    for ip in img_paths:
        lp = find_label(labels_dir, ip)
        if not lp.exists():
            continue
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            stats["unreadable_image"] += 1
            continue
        W, H = im.size

        persons, ppes = read_yolo_label(lp, W, H)
        if not persons:
            stats["image_no_person"] += 1
            continue

        tallies = assign_ppe_to_persons(
            persons, ppes, use_spatial_prior=not args.no_spatial_prior
        )

        # đếm PPE không gán được cho ai (để biết chất lượng thuật toán)
        assigned = sum(
            t[g]["pos"] + t[g]["neg"] for t in tallies for g in GROUP_NAMES
        )
        stats["ppe_total"] += len(ppes)
        stats["ppe_assigned"] += assigned

        for idx, pbox in enumerate(persons):
            ex = expand_box(pbox, CROP_MARGIN, W, H)
            cw, ch = ex[2] - ex[0], ex[3] - ex[1]
            if min(cw, ch) < args.min_side:
                stats["crop_too_small"] += 1
                continue

            n_overlap = sum(
                1 for j, other in enumerate(persons)
                if j != idx and iou(ex, other) > OVERLAP_IOU
            )

            labels = tally_to_labels(tallies[idx])
            for g, v in zip(GROUP_NAMES, labels):
                stats[f"label_{g}_{ {0:'ok', 1:'violation', -1:'unknown'}[v] }"] += 1

            name = f"{ip.stem}_p{idx}.jpg"
            im.crop((int(ex[0]), int(ex[1]), int(ex[2]), int(ex[3]))).save(
                crop_dir / name, quality=92
            )
            writer.writerow(
                [f"{split}/{name}", split, ip.name, idx,
                 round(ex[0], 1), round(ex[1], 1), round(ex[2], 1), round(ex[3], 1),
                 *labels, n_overlap, int(cw), int(ch)]
            )
            stats["crops"] += 1


def resolve_splits(args):
    """
    Trả về {split: (img_dir, lbl_dir)}.
    Ưu tiên --train-dir/--val-dir/--test-dir; nếu không có thì dùng
    bố cục chuẩn dưới --src.
    """
    out = {}
    explicit = {"train": args.train_dir, "val": args.val_dir, "test": args.test_dir}
    for split, d in explicit.items():
        if d:
            base = Path(d)
            out[split] = (base / "images", base / "labels")
    if out:
        return out
    src = Path(args.src)
    for split in SPLITS:
        out[split] = (src / "images" / split, src / "labels" / split)
    return out


def load_class_names(args, splits):
    """Tìm data.yaml và nạp thứ tự lớp thật. Cảnh báo nếu các file không khớp."""
    candidates = []
    if args.data_yaml:
        candidates.append(Path(args.data_yaml))
    else:
        seen = set()
        for img_dir, _ in splits.values():
            # data.yaml thường nằm cùng cấp hoặc cao hơn thư mục split
            for p in (img_dir.parent, img_dir.parent.parent, img_dir.parent.parent.parent):
                y = p / "data.yaml"
                if y.exists() and y.resolve() not in seen:
                    seen.add(y.resolve())
                    candidates.append(y)

    if not candidates:
        print("⚠ Không tìm thấy data.yaml — dùng thứ tự lớp mặc định trong common.py")
        print(f"  {common.CLASSES}")
        return

    found = {}
    for y in candidates:
        try:
            found[str(y)] = read_names_from_yaml(y)
        except Exception as e:
            print(f"⚠ bỏ qua {y}: {e}")

    if not found:
        print("⚠ Không đọc được data.yaml nào — dùng thứ tự mặc định")
        return

    uniq = {tuple(v) for v in found.values()}
    if len(uniq) > 1:
        print("\n" + "!" * 62)
        print("  LỖI: các data.yaml có thứ tự lớp KHÁC NHAU")
        for f, n in found.items():
            print(f"    {f}\n      {n}")
        print("  Nhãn train và test sẽ lệch nhau. Sửa trước khi chạy tiếp.")
        print("!" * 62)
        raise SystemExit(2)

    names = next(iter(found.values()))
    print(f"Thứ tự lớp đọc từ {list(found)[0]}:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")
    for w in configure_classes(names):
        print(f"⚠ {w}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="dataset gốc theo bố cục images/labels/{split}")
    ap.add_argument("--train-dir", help="thư mục chứa images/ và labels/ của train")
    ap.add_argument("--val-dir", help="tương tự cho val")
    ap.add_argument("--test-dir", help="tương tự cho test")
    ap.add_argument("--data-yaml", help="chỉ định data.yaml để lấy thứ tự lớp")
    ap.add_argument("--out", required=True, help="thư mục xuất")
    ap.add_argument("--min-side", type=int, default=MIN_CROP_SIDE)
    ap.add_argument("--no-spatial-prior", action="store_true",
                    help="tắt tiên nghiệm vị trí, chỉ dùng IoA")
    ap.add_argument("--skip-phase1", action="store_true")
    args = ap.parse_args()

    if not args.src and not any([args.train_dir, args.val_dir, args.test_dir]):
        ap.error("cần --src, hoặc ít nhất một trong --train-dir/--val-dir/--test-dir")

    splits = resolve_splits(args)
    load_class_names(args, splits)

    out = Path(args.out)
    (out / "phase2").mkdir(parents=True, exist_ok=True)

    manifest = out / "phase2" / "manifest.csv"
    stats = Counter()
    p1_summary = {}

    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "split", "src_image", "person_idx",
                    "x1", "y1", "x2", "y2",
                    *GROUP_NAMES, "n_overlap_persons", "crop_w", "crop_h"])

        for split in SPLITS:
            if split not in splits:
                continue
            img_dir, lbl_dir = splits[split]
            if not img_dir.exists():
                print(f"  bỏ qua split '{split}' (không tìm thấy {img_dir})")
                continue
            imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            print(f"[{split}] {len(imgs)} ảnh  ({img_dir})")

            if not args.skip_phase1:
                p1_summary[split] = build_phase1(out, split, imgs, lbl_dir)

            build_phase2(out, split, imgs, lbl_dir, args, w, stats)

    # data.yaml cho pha 1
    if not args.skip_phase1 and p1_summary:
        (out / "phase1").mkdir(parents=True, exist_ok=True)
        (out / "phase1" / "data.yaml").write_text(
            f"path: {(out / 'phase1').resolve()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "nc: 1\nnames: [person]\n"
        )

    if not stats["crops"]:
        print("\n⚠ Không sinh được crop nào — kiểm tra lại đường dẫn split.")
        raise SystemExit(1)

    # ---------------------------------------------------------- báo cáo
    print("\n" + "=" * 58)
    print("  THỐNG KÊ")
    print("=" * 58)
    if p1_summary:
        for s, (ni, nb) in p1_summary.items():
            print(f"pha 1  {s:5s}: {ni:5d} ảnh, {nb:6d} box người")
    print(f"\npha 2  tổng crop        : {stats['crops']}")
    print(f"       crop bị loại (nhỏ): {stats['crop_too_small']}")
    if stats["ppe_total"]:
        rate = stats["ppe_assigned"] / stats["ppe_total"] * 100
        print(f"       PPE gán được      : {stats['ppe_assigned']}/{stats['ppe_total']}"
              f"  ({rate:.1f}%)")
        if rate < 85:
            print("       ⚠ tỉ lệ gán thấp — xem lại IOA_THRESH / y_range trong common.py")

    print(f"\n{'nhóm':<10}{'tuân thủ':>12}{'vi phạm':>12}{'không rõ':>12}{'% không rõ':>12}")
    for g in GROUP_NAMES:
        ok = stats[f"label_{g}_ok"]
        vi = stats[f"label_{g}_violation"]
        un = stats[f"label_{g}_unknown"]
        tot = ok + vi + un
        pct = un / tot * 100 if tot else 0
        flag = "  ⚠" if pct > 50 else ""
        print(f"{g:<10}{ok:>12}{vi:>12}{un:>12}{pct:>11.1f}%{flag}")

    (out / "phase2" / "stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(f"\nmanifest: {manifest}")
    print("→ Mở ngẫu nhiên ~100 crop đối chiếu nhãn bằng mắt TRƯỚC KHI train.")


if __name__ == "__main__":
    main()
