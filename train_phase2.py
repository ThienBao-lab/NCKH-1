"""
Pha 2 — bộ phân loại đa nhãn trạng thái vi phạm PPE trên ảnh crop người.

4 đầu ra sigmoid: helmet / vest / gloves / boots.
Nhãn -1 (không xác định) bị mask khỏi loss và khỏi metric.

Chạy:
    python train_phase2.py --root OUT/phase2
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import CROP_H, CROP_W, GROUP_NAMES, N_GROUPS


# ---------------------------------------------------------------- dataset
class CropDataset(Dataset):
    def __init__(self, root, split, tf, max_overlap=None):
        self.root, self.tf = Path(root), tf
        self.rows = []
        with (Path(root) / "manifest.csv").open() as f:
            for r in csv.DictReader(f):
                if r["split"] != split:
                    continue
                if max_overlap is not None and int(r["n_overlap_persons"]) > max_overlap:
                    continue
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / "crops" / r["crop"]).convert("RGB")
        y = torch.tensor([float(r[g]) for g in GROUP_NAMES])
        mask = (y >= 0).float()          # 1 nếu nhãn xác định
        return self.tf(img), y.clamp(min=0), mask


def build_transforms(train):
    # KHÔNG dùng lật dọc / xoay lớn: sẽ phá tiên nghiệm "mũ trên, giày dưới"
    base = [transforms.Resize((CROP_H, CROP_W))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if train:
        base.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    return transforms.Compose(base)


# ---------------------------------------------------------------- model
def build_model(arch, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, N_GROUPS)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, N_GROUPS)
    elif arch == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, N_GROUPS)
    else:
        raise ValueError(f"arch không hỗ trợ: {arch}")
    return m


def compute_pos_weight(ds):
    """Cân bằng lớp: nhóm nào ít mẫu vi phạm thì được nhân trọng số lên."""
    pos = torch.zeros(N_GROUPS)
    neg = torch.zeros(N_GROUPS)
    for r in ds.rows:
        for k, g in enumerate(GROUP_NAMES):
            v = int(r[g])
            if v == 1:
                pos[k] += 1
            elif v == 0:
                neg[k] += 1
    return (neg / pos.clamp(min=1)).clamp(0.2, 20.0)


# ---------------------------------------------------------------- metric
@torch.no_grad()
def evaluate(model, loader, device, thresh=0.5, amp=False):
    model.eval()
    tp = torch.zeros(N_GROUPS)
    fp = torch.zeros(N_GROUPS)
    fn = torch.zeros(N_GROUPS)
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        p = (torch.sigmoid(logits.float()).cpu() > thresh).float()
        tp += ((p == 1) & (y == 1) & (mask == 1)).sum(0)
        fp += ((p == 1) & (y == 0) & (mask == 1)).sum(0)
        fn += ((p == 0) & (y == 1) & (mask == 1)).sum(0)
    prec = tp / (tp + fp).clamp(min=1e-9)
    rec = tp / (tp + fn).clamp(min=1e-9)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return prec, rec, f1


def print_table(prec, rec, f1):
    print(f"{'nhóm':<10}{'P':>10}{'R':>10}{'F1':>10}")
    for k, g in enumerate(GROUP_NAMES):
        print(f"{g:<10}{prec[k]:>10.4f}{rec[k]:>10.4f}{f1[k]:>10.4f}")
    print(f"{'macro':<10}{prec.mean():>10.4f}{rec.mean():>10.4f}{f1.mean():>10.4f}")


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục OUT/phase2")
    ap.add_argument("--arch", default="convnext_tiny",
                    choices=["convnext_tiny", "efficientnet_b0", "resnet50"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None,
                    help="mặc định tự scale tuyến tính từ 3e-4 @ batch 64")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-overlap", type=int, default=None,
                    help="lọc crop có nhiều hơn N người khác lọt vào")
    ap.add_argument("--no-amp", action="store_true",
                    help="tắt bfloat16 autocast")
    ap.add_argument("--out", default="runs_ppe2/phase2")
    args = ap.parse_args()

    if args.lr is None:                       # linear scaling rule
        args.lr = 3e-4 * args.batch / 64
        print(f"lr tự chọn: {args.lr:.2e}  (batch {args.batch})")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not args.no_amp) and device == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = True
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
              f"| torch {torch.__version__} cuda {torch.version.cuda} | bf16={amp}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tr = CropDataset(args.root, "train", build_transforms(True), args.max_overlap)
    va = CropDataset(args.root, "val", build_transforms(False))
    te = CropDataset(args.root, "test", build_transforms(False))
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} crop")

    def dl(ds, sh):
        kw = dict(batch_size=args.batch, shuffle=sh,
                  num_workers=args.workers, pin_memory=True)
        if args.workers > 0:
            kw["prefetch_factor"] = 4
            if sh:                      # chỉ giữ worker sống cho loader train
                kw["persistent_workers"] = True
        return DataLoader(ds, **kw)

    tr_dl, va_dl, te_dl = dl(tr, True), dl(va, False), dl(te, False)

    pw = compute_pos_weight(tr)
    print("pos_weight:", {g: round(float(v), 2) for g, v in zip(GROUP_NAMES, pw)})

    model = build_model(args.arch).to(device).to(memory_format=torch.channels_last)
    crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, mask in tr_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss_raw = crit(model(x), y)
                loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        p, r, f1 = evaluate(model, va_dl, device, amp=amp)
        macro = float(f1.mean())
        history.append({"epoch": ep, "loss": total / n, "val_macro_f1": macro})
        print(f"ep {ep:3d}  loss {total/n:.4f}  val macro-F1 {macro:.4f}")

        if macro > best_f1:
            best_f1 = macro
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "groups": GROUP_NAMES}, outdir / "best.pt")

    print("\n" + "=" * 44)
    print(f"  TEST (Protocol 1 — oracle, crop từ box GT)")
    print("=" * 44)
    model.load_state_dict(torch.load(outdir / "best.pt", map_location=device)["model"])
    model.to(memory_format=torch.channels_last)
    print_table(*evaluate(model, te_dl, device, amp=amp))

    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\ncheckpoint: {outdir/'best.pt'}")
    print("Đây là TRẦN TRÊN của pha 2. Số thật phải chạy evaluate.py (end-to-end).")


if __name__ == "__main__":
    main()