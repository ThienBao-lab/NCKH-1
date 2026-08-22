# Kết quả lần 2 — Pipeline 2 pha phát hiện vi phạm an toàn lao động

Ngày đo: **2026-08-22** · Tập test: **426 ảnh / 1741 người** · **3 seed cho cả hai hệ**

Lần 1 xem `KETQUA.md`. Lần này bổ sung: model 1 pha để so sánh, 3 seed hai phía,
ngưỡng riêng từng nhóm, và huấn luyện pha 1 thay cho zero-shot.

| pha | model | huấn luyện |
|---|---|---|
| Pha 1 | `yolov8m` 1 lớp person | 50 epoch, imgsz 640, batch 64 |
| Pha 2 | `convnext_tiny` (27,8M) | 40 epoch, batch 96, lr 1e-4, bf16 |
| 1 pha (đối chứng) | `yolov8m` 9 lớp | 50 epoch, imgsz 640, batch 64 |

Ngưỡng vận hành pha 1: **conf = 0,55** — xem mục 2.
Cả 3 seed của mỗi hệ dùng **chung một bộ siêu tham số**, chỉ khác `--seed`.

---

## 1. Thống kê dữ liệu (Bảng 1)

| chỉ số | giá trị |
|---|---|
| Tổng crop | **6734** |
| — train / val / test | 4629 (68,7%) / 509 (7,6%) / 1596 (23,7%) |
| PPE gán được cho người | 18552 / 19783 = **93,78%** |
| PPE không gán được | 1231 (6,22%) |
| Crop bị loại vì quá nhỏ | 371 (5,22% số crop sinh ra) |
| Ảnh không có người | 301 |

### Tỉ lệ nhãn không xác định

| nhóm | tuân thủ | vi phạm | không xác định | % không xác định |
|---|---|---|---|---|
| helmet | 3634 | 2105 | 995 | 14,8% |
| vest | 3555 | 2205 | 974 | 14,5% |
| gloves | 1140 | 1142 | 4452 | **66,1%** |
| boots | 1834 | 425 | 4475 | **66,5%** |

Trên tập test, số ca **xác định được** (mẫu thực sự vào phép đếm):
helmet 1282 · vest 1328 · gloves 701 · **boots 576** (trên 1596 crop).
Số ca vi phạm: helmet 694 · vest 747 · gloves 368 · **boots 117**.

> **Hạn chế dữ liệu, không sửa được bằng model.** Nghi ngờ ban đầu là ảnh
> thường cắt ngang thân người nên không thấy chân — đúng, boots 66,5% không
> xác định. Nhưng gloves cũng 66,1%, nên đây không phải vấn đề riêng của boots.
> Với boots còn một nguyên nhân thứ hai: chỉ 117 ca vi phạm trên 576 ca xác
> định được (20,3% dương). Lớp vừa ít mẫu vừa lệch nặng, nên σ giữa các seed ở
> nhóm boots lớn gấp ~11 lần helmet. Con số F1 thấp của boots phần lớn là giới
> hạn thống kê chứ không chỉ là model yếu.

---

## 2. Chọn ngưỡng conf của pha 1 (con số phải ghi vào bài: **0,55**)

Quét trên tập **val**, hai chỉ tiêu khác nhau cho hai kết quả khác nhau:

| conf | Recall **riêng pha 1** | macro-F1 **end-to-end** |
|---|---|---|
| 0,20 | 0,9002 | 0,7446 |
| 0,30 | **0,9021** ← đỉnh | 0,7498 |
| 0,40 | 0,8738 | 0,7800 |
| 0,50 | 0,8531 | 0,7953 |
| **0,55** | 0,8343 | **0,7981** ← đỉnh |
| 0,65 | — | 0,7920 |
| 0,70 | — | 0,7913 |

> **Tiền đề "ưu tiên Recall hơn Precision" bị dữ liệu bác bỏ.** Lập luận ban đầu
> là "crop thừa còn được pha 2 lọc, người bỏ sót thì không cứu được". Vế đầu
> không đúng: pha 2 chỉ phân loại PPE, nó **không có lớp "đây không phải người"**,
> nên mỗi box thừa mà pha 2 gán vi phạm trở thành FP của cả hệ. Hạ conf từ 0,55
> xuống 0,30 đúng là tăng recall pha 1 (0,8343 → 0,9021) nhưng macro-F1 toàn hệ
> **giảm** (0,7981 → 0,7498).
>
> Ngưỡng tối ưu cho pha 1 *riêng lẻ* (0,30) và cho *toàn hệ* (0,55) là hai điểm
> khác nhau — xem `figures/fig_conf_e2e.pdf`. Ngưỡng 0,55 của lần 1 là đúng.

Ngưỡng được chọn trên **val**, không phải test.

---

## 3. Pha 2 đánh giá riêng — kết quả oracle (Bảng 3, trần trên)

Crop cắt từ box người có sẵn trong nhãn; pha 1 không tham gia.

| nhóm | seed 0 | seed 1 | seed 2 | **mean ± σ** |
|---|---|---|---|---|
| helmet | 0,9090 | 0,9077 | 0,9014 | **0,9060 ± 0,0041** |
| vest | 0,9189 | 0,8832 | 0,9025 | **0,9015 ± 0,0179** |
| gloves | 0,7242 | 0,7208 | 0,6818 | **0,7089 ± 0,0236** |
| boots | 0,5625 | 0,4951 | 0,4779 | **0,5118 ± 0,0447** |
| **macro** | 0,7787 | 0,7517 | 0,7409 | **0,7571 ± 0,0195** |

Lần 1 đo được macro 0,7688 với một lần chạy duy nhất — nằm trong khoảng
0,7571 ± 0,0195, nên không phải hồi quy; chỉ là lần 1 không có σ để biết mình
đang ở đâu trong phân bố.

---

## 4. Bảng chính — 2 pha vs 1 pha (Bảng 2)

Cấu hình chính thức: pha 1 fine-tune @ conf 0,55 + ngưỡng riêng từng nhóm.
Mọi ô là **mean ± σ trên 3 seed**. Model 1 pha được quy về "vi phạm theo từng
người" bằng đúng thuật toán gán đã dùng lúc sinh dữ liệu.

| nhóm | **2 pha F1** | **1 pha F1** | chênh |
|---|---|---|---|
| helmet | 0,7189 ± 0,0032 | **0,7458 ± 0,0236** | −0,0269 |
| vest | 0,7080 ± 0,0111 | **0,7249 ± 0,0109** | −0,0169 |
| gloves | **0,6391 ± 0,0296** | 0,4658 ± 0,0156 | **+0,1733** |
| boots | **0,4530 ± 0,0296** | 0,2934 ± 0,0434 | **+0,1596** |
| **macro** | **0,6298 ± 0,0157** | 0,5574 ± 0,0152 | **+0,0724** |

Tách theo P và R:

| | 2 pha | 1 pha |
|---|---|---|
| macro P | 0,7667 ± 0,0119 | 0,7505 ± 0,0291 |
| macro R | **0,5380 ± 0,0197** | 0,4510 ± 0,0133 |

> **Toàn bộ lợi thế của 2 pha nằm ở vật nhỏ.** Ở helmet và vest, model 1 pha
> thắng nhẹ — nó nhìn cả ảnh nên bắt mũ và áo phản quang dễ như nhau, còn 2 pha
> phải trả giá cho lỗi pha 1. Ở gloves và boots, 2 pha hơn ~17 và ~16 điểm F1,
> vì crop phóng to vùng người mới đủ độ phân giải cho vật nhỏ. Chênh lệch đến
> gần như hoàn toàn từ **Recall** (+0,087) chứ không phải Precision (+0,016):
> model 1 pha đơn giản là không nhìn thấy găng tay và giày.

---

## 5. Ngưỡng chung 0,5 vs ngưỡng riêng từng nhóm (Bảng 4)

Ngưỡng chọn trên val, áp lên test. Ngưỡng chọn được:
helmet 0,60 · vest 0,30 · gloves 0,50 · boots 0,50.

**Oracle (pha 2 riêng, seed 0):**

| nhóm | F1 @ 0,5 | F1 ngưỡng riêng | chênh |
|---|---|---|---|
| helmet | 0,9090 | 0,9072 | −0,0018 |
| vest | 0,9189 | 0,9259 | +0,0070 |
| gloves | 0,7242 | 0,7242 | 0 |
| boots | 0,5625 | 0,5625 | 0 |
| **macro** | 0,7787 | 0,7800 | **+0,0013** |

**End-to-end (3 seed):** macro-F1 0,6283 ± 0,0164 → 0,6298 ± 0,0157,
chênh **+0,0015** — nhỏ hơn σ một bậc, tức **không có ý nghĩa thống kê**.

> **Kết quả âm, đáng ghi vào bài.** Giả thuyết là "khoảng cách AP−F1 lớn thì
> chỉnh ngưỡng sẽ ăn". Sai: helmet có khoảng cách AP−F1 lớn nhất lại là nhóm
> duy nhất **tệ đi**. gloves và boots giữ nguyên 0,50 vì grid search trên val
> không tìm được gì tốt hơn. Nguyên nhân là val chỉ 509 crop — quá nhỏ để chọn
> ngưỡng đáng tin.

---

## 6. Pha 1: fine-tune hay zero-shot (phân tích domain shift)

Tập test đến từ bundle Roboflow **khác** với train/val (`test/ <- test_fixed/valid`
theo README dataset). Hệ quả:

| pha 1 | R trên **val** | R trên **test** | macro-F1 e2e (test, 3 seed) |
|---|---|---|---|
| yolov8m fine-tune @0,55 | **0,9021** @0,30 | 0,5744 | 0,6298 ± 0,0157 |
| yolo26m zero-shot @0,70 | 0,8512 @0,30 | — | **0,6423 ± 0,0133** |

> **Fine-tune pha 1 thắng rõ trên val nhưng thua trên test.** 1838 ảnh train
> không đủ để vượt COCO pretrain khi test đổi domain. Cấu hình chính vẫn lấy
> bản fine-tune vì nó được chọn theo val — đổi sang zero-shot lúc này là chọn
> model trên chính tập đánh giá.
>
> Kết luận cho bài: **khoảng cách val–test là hạn chế của thiết kế tập dữ liệu**,
> lặp lại ở cả ba chỗ độc lập — pha 1 (R 0,90 val / 0,57 test), ngưỡng nhóm
> (boots F1 0,9091 val / 0,5625 test), và ngưỡng conf (val nói 0,55 > 0,30,
> test nói ngược lại). Mọi kết luận rút từ val trong bài này cần đọc kèm cảnh báo đó.

---

## 7. Những chỗ lần 2 không bằng lần 1

1. **macro-F1 end-to-end thấp hơn**: 0,6298 ± 0,0157 (lần 2) so với 0,6641
   (lần 1, một seed). Nguyên nhân là pha 1 fine-tune kém hơn `yolo26m` zero-shot
   trên test — xem mục 6. Cấu hình zero-shot của lần 2 (0,6423 ± 0,0133) cũng
   chưa bằng 0,6641 của lần 1, phần còn lại là dao động giữa seed.
2. **Mặc định `train_phase2.py` bị hỏng.** Commit mới đổi mặc định thành
   `batch 512` + lr tự scale → **2,4e-3**, cao gấp 24 lần lr 1e-4 của lần 1.
   Với bộ đó loss đứng yên ở 0,92 sau 12 epoch (lần 1: 0,038) và pha 2 gần như
   không học. Batch 512 trên 4629 crop cũng chỉ còn 9 bước/epoch thay vì 48.
   **Đã chạy bằng `--batch 96 --lr 1e-4`**, cố định chung cho cả 3 seed.
   Comment giải thích điều này trong `train_phase2.py` đã bị xóa ở commit mới —
   nên khôi phục mặc định cũ.
3. **`export_examples.py` vẽ chữ không đọc được — đã sửa.** Script dùng font
   bitmap mặc định của PIL, không có glyph cho tiếng Việt (`mũ`, `áo`, `găng`,
   `giày`) lẫn `✓`/`✗`, nên nhãn ra thành ký tự rác; font đó cũng không phóng to
   theo `--scale`. Đã chuyển sang DejaVuSans-Bold, cỡ chữ theo cạnh ảnh, kèm cơ
   chế đẩy nhãn tránh chồng khi nhiều người đứng sát nhau.
4. **`export_examples.py` không nhận `--thresholds`**, nên ảnh minh hoạ dùng
   ngưỡng 0,5 chung, lệch nhẹ so với cấu hình chính thức. Ảnh hưởng không đáng
   kể vì ngưỡng riêng chỉ đổi +0,0015.

---

## 8. Hình

| file | nội dung |
|---|---|
| `fig_per_group` | P/R/F1 từng nhóm, 2 pha vs 1 pha, thanh sai số 3 seed |
| `fig_macro` | Macro P/R/F1 tổng |
| `fig_conf_sweep` | P/R/F1 của riêng pha 1 theo conf |
| **`fig_conf_e2e`** | **Ngưỡng tối ưu pha 1 riêng lẻ ≠ ngưỡng tối ưu toàn hệ** (mới) |
| `fig_thresholds` | F1 trước/sau khi tối ưu ngưỡng riêng |
| `fig_label_stats` | Phân bố nhãn tuân thủ / vi phạm / không xác định |
| `figures/examples/` | 12 ảnh minh hoạ, ưu tiên ảnh nhiều người + vi phạm vật nhỏ |

---

## 9. Tái lập

```bash
# Pha 1 — bộ phát hiện người
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_phase1.py \
    --data data2phase/phase1/data.yaml --model yolov8m.pt --seed 0

# Model 1 pha đối chứng, 3 seed
for S in 0 1 2; do .venv/bin/python train_baseline.py \
    --data dataset/data.yaml --model yolov8m.pt --seed $S; done

# Pha 2, 3 seed, CÙNG một bộ siêu tham số
for S in 0 1 2; do .venv/bin/python train_phase2.py --root data2phase/phase2 \
    --epochs 40 --batch 96 --lr 1e-4 --seed $S --out runs_ppe2/phase2_s$S; done

# Chọn ngưỡng conf theo chỉ tiêu end-to-end, trên val
.venv/bin/python sweep_conf_e2e.py --phase1 runs_ppe2/phase1_person/weights/best.pt \
    --phase2 runs_ppe2/phase2_s0/best.pt \
    --out runs_ppe2/phase1_person/conf_sweep_e2e_val.csv

# Ngưỡng riêng từng nhóm
.venv/bin/python tune_thresholds.py --root data2phase/phase2 \
    --ckpt runs_ppe2/phase2_s0/best.pt --out runs_ppe2/thresholds.json

# Bảng chính
for S in 0 1 2; do .venv/bin/python evaluate.py --src dataset --split test \
    --data-yaml dataset/data.yaml --phase1 runs_ppe2/phase1_person/weights/best.pt \
    --phase2 runs_ppe2/phase2_s$S/best.pt --conf1 0.55 \
    --thresholds runs_ppe2/thresholds.json \
    --baseline runs_ppe2/baseline1_seed$S/weights/best.pt \
    --save-json runs_ppe2/eval_seed$S.json; done

.venv/bin/python summarize_seeds.py --pattern 'runs_ppe2/eval_seed{}.json'
.venv/bin/python make_figures.py --results runs_ppe2 \
    --stats data2phase/phase2/stats.json \
    --conf-sweep runs_ppe2/phase1_person/conf_sweep_val.csv --out figures
.venv/bin/python fig_conf_e2e.py
```

Log: `runs_ppe2/logs/` · Checkpoint pha 2: `runs_ppe2/phase2_s{0,1,2}/best.pt`
