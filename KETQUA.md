# Kết quả — Pipeline 2 pha phát hiện vi phạm an toàn lao động

Ngày đo: **2026-08-19** · Tập test: **426 ảnh / 1741 người**

| pha | model | huấn luyện |
|---|---|---|
| Pha 1 | `yolo26m` zero-shot COCO | không train |
| Pha 2 | `convnext_tiny` (27,8M) | 40 epoch, batch 96, lr 1e-4, bf16 |

Ngưỡng vận hành pha 1: **conf = 0,55**

---

## 1. Pha 1 — phát hiện người

| chỉ số | giá trị |
|---|---|
| **mAP50** | **0,8405** |
| mAP50-95 | 0,4777 |

| TP | FP | FN | **P** | **R** | **F1** |
|---|---|---|---|---|---|
| 1175 | 112 | 566 | **0,9130** | **0,6749** | **0,7761** |

---

## 2. Pha 2 — ConvNeXt-tiny, đánh giá riêng

Crop cắt từ box người có sẵn trong nhãn (pha 1 không tham gia).

| lớp | **P** | **R** | **F1** | **AP** |
|---|---|---|---|---|
| helmet | 0,9495 | 0,8674 | **0,9066** | 0,9709 |
| vest | 0,9526 | 0,8608 | **0,9044** | 0,9777 |
| gloves | 0,7563 | 0,6495 | 0,6988 | 0,7982 |
| boots | 0,5583 | 0,5726 | 0,5654 | 0,5721 |
| **macro** | **0,8042** | **0,7376** | **0,7688** | **0,8297** |

---

## 3. Kết quả end-to-end (ghép 2 pha)

Crop cắt từ box `yolo26m` tự đoán. Người pha 1 bỏ sót tính vào FN,
box thừa không khớp ai mà bị kêu vi phạm tính vào FP.

| lớp | **P** | **R** | **F1** | số ca vi phạm |
|---|---|---|---|---|
| helmet | 0,8952 | 0,7297 | **0,8040** | 714 |
| vest | 0,8955 | 0,6750 | **0,7698** | 800 |
| gloves | 0,6443 | 0,5973 | 0,6199 | 370 |
| boots | 0,4275 | 0,5043 | 0,4627 | 117 |
| **macro** | **0,7156** | **0,6266** | **0,6641** | 1541 |

### Cái giá của việc ghép

| lớp | F1 riêng | F1 ghép | mất |
|---|---|---|---|
| helmet | 0,9066 | 0,8040 | −0,1026 |
| vest | 0,9044 | 0,7698 | −0,1346 |
| gloves | 0,6988 | 0,6199 | −0,0789 |
| boots | 0,5654 | 0,4627 | −0,1027 |
| **macro** | **0,7688** | **0,6641** | **−0,1047** |

Mất 10,5 điểm macro-F1, chủ yếu ở Recall (0,7376 → 0,6266) chứ không phải
Precision (0,8042 → 0,7156): 566 người pha 1 không tìm thấy trở thành FN.

---

## 4. Tái lập

```bash
uv venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

.venv/bin/python prepare_data.py --src dataset --out data2phase

CUDA_VISIBLE_DEVICES=1 .venv/bin/python train_phase2.py \
    --root data2phase/phase2 --arch convnext_tiny \
    --epochs 40 --batch 96 --lr 1e-4 --out runs_ppe2/phase2_convnext

.venv/bin/python evaluate.py --src dataset --split test \
    --phase1 yolo26m.pt --phase2 runs_ppe2/phase2_convnext/best.pt \
    --data-yaml dataset/data.yaml --conf1 0.55
```

Log: `runs_ppe2/logs/` · Checkpoint pha 2: `runs_ppe2/phase2_convnext/best.pt`
