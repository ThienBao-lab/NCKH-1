"""
Tối ưu ngưỡng quyết định riêng cho từng nhóm PPE ở pha 2.

Mặc định pha 2 dùng chung ngưỡng 0.5 cho cả 4 nhóm. Nhưng khoảng cách giữa
AP và F1 cho thấy 0.5 chưa tối ưu với vài nhóm — chỉnh ngưỡng là cải thiện
KHÔNG cần train lại.

Quy trình (thứ tự này quan trọng về mặt phương pháp luận):
  1. Quét ngưỡng trên tập VALIDATION, chọn mức cho F1 cao nhất mỗi nhóm
  2. Áp ngưỡng đã chọn lên tập TEST, báo cáo kết quả

Chọn ngưỡng trực tiếp trên test là tối ưu trên chính tập đánh giá — kết quả
sẽ lạc quan giả tạo và phản biện sẽ bắt lỗi ngay.

Chạy:
    python tune_thresholds.py --root OUT/phase2 --ckpt runs_ppe2/phase2/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import GROUP_NAMES, N_GROUPS
from train_phase2 import CropDataset, build_model, build_transforms


# ---------------------------------------------------------------- thu logits
@torch.no_grad()
def collect_scores(model, loader, device, amp=False):
    """Chạy model một lượt, trả về (xác suất, nhãn, mask) cho cả tập."""
    model.eval()
    probs, ys, masks = [], [], []
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y)
        masks.append(mask)
    return torch.cat(probs), torch.cat(ys), torch.cat(masks)


def prf_at(prob, y, mask, k, thresh):
    """P, R, F1 của nhóm thứ k tại một ngưỡng."""
    m = mask[:, k] == 1
    if m.sum() == 0:
        return 0.0, 0.0, 0.0
    pred = (prob[m, k] > thresh).float()
    gt = y[m, k]
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục OUT/phase2")
    ap.add_argument("--ckpt", required=True, help="best.pt của pha 2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs_ppe2/thresholds.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not args.no_amp) and device == "cuda" and torch.cuda.is_bf16_supported()

    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).to(memory_format=torch.channels_last)

    tf = build_transforms(False)
    def dl(split):
        ds = CropDataset(args.root, split, tf)
        kw = dict(batch_size=args.batch, shuffle=False,
                  num_workers=args.workers, pin_memory=True)
        return DataLoader(ds, **kw), len(ds)

    va_dl, n_va = dl("val")
    te_dl, n_te = dl("test")
    print(f"val {n_va} crop | test {n_te} crop")
    if n_va == 0:
        raise SystemExit("Tập val rỗng — không chọn ngưỡng được.")

    print("\nChạy model trên val và test (mỗi tập một lượt)...")
    va_prob, va_y, va_mask = collect_scores(model, va_dl, device, amp)
    te_prob, te_y, te_mask = collect_scores(model, te_dl, device, amp)

    # ---------- bước 1: chọn ngưỡng trên VAL ----------
    # round() trước khi int(): (0.70-0.30)/0.02 ra 19.999... nếu không sẽ
    # bị cắt mất mức cuối
    n_step = int(round((args.hi - args.lo) / args.step)) + 1
    grid = [round(args.lo + args.step * i, 4) for i in range(n_step)]

    best = {}
    print("\n" + "=" * 62)
    print("  BƯỚC 1 — Chọn ngưỡng trên tập VALIDATION")
    print("=" * 62)
    print(f"{'nhóm':<10}{'ngưỡng':>9}{'F1@val':>10}{'F1@0.5':>10}{'chênh':>10}")
    for k, g in enumerate(GROUP_NAMES):
        _, _, f1_default = prf_at(va_prob, va_y, va_mask, k, 0.5)
        best_t, best_f = 0.5, f1_default
        for t in grid:
            _, _, f = prf_at(va_prob, va_y, va_mask, k, t)
            if f > best_f:
                best_t, best_f = t, f
        best[g] = best_t
        print(f"{g:<10}{best_t:>9.2f}{best_f:>10.4f}{f1_default:>10.4f}"
              f"{best_f - f1_default:>+10.4f}")

    # ---------- bước 2: áp lên TEST ----------
    print("\n" + "=" * 62)
    print("  BƯỚC 2 — Áp ngưỡng đã chọn lên tập TEST")
    print("=" * 62)
    print(f"{'nhóm':<10}{'ngưỡng':>8}"
          f"{'P@0.5':>9}{'R@0.5':>9}{'F1@0.5':>9}"
          f"{'P_moi':>9}{'R_moi':>9}{'F1_moi':>9}{'chênh':>9}")

    sum_old = sum_new = 0.0
    rows = []
    for k, g in enumerate(GROUP_NAMES):
        p0, r0, f0 = prf_at(te_prob, te_y, te_mask, k, 0.5)
        p1, r1, f1 = prf_at(te_prob, te_y, te_mask, k, best[g])
        sum_old += f0
        sum_new += f1
        rows.append((g, best[g], p0, r0, f0, p1, r1, f1))
        print(f"{g:<10}{best[g]:>8.2f}"
              f"{p0:>9.4f}{r0:>9.4f}{f0:>9.4f}"
              f"{p1:>9.4f}{r1:>9.4f}{f1:>9.4f}{f1 - f0:>+9.4f}")

    macro_old, macro_new = sum_old / N_GROUPS, sum_new / N_GROUPS
    print("-" * 62)
    print(f"{'macro-F1':<18}{macro_old:>27.4f}{macro_new:>27.4f}"[:62])
    print(f"\nmacro-F1:  0.5 chung {macro_old:.4f}  ->  ngưỡng riêng {macro_new:.4f}"
          f"   ({macro_new - macro_old:+.4f})")

    if macro_new <= macro_old:
        print("\nNgưỡng riêng KHÔNG tốt hơn trên test — ngưỡng chọn từ val không")
        print("tổng quát được. Cứ dùng 0.5 và ghi lại kết quả này trong bài.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "thresholds": best,
        "macro_f1_test_default": round(macro_old, 4),
        "macro_f1_test_tuned": round(macro_new, 4),
        "per_group_test": [
            {"group": g, "threshold": t,
             "P_default": round(p0, 4), "R_default": round(r0, 4), "F1_default": round(f0, 4),
             "P_tuned": round(p1, 4), "R_tuned": round(r1, 4), "F1_tuned": round(f1, 4)}
            for g, t, p0, r0, f0, p1, r1, f1 in rows
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
