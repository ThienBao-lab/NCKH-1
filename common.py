"""
Cấu hình dùng chung + logic gán PPE cho người.

Quy ước nhãn cho pha 2 (mỗi người, mỗi nhóm PPE):
    0  = tuân thủ   (compliant)
    1  = vi phạm    (violation)
   -1  = không xác định (undetermined) -> bị mask khỏi loss, bỏ khỏi metric
"""

from pathlib import Path

# ---------------------------------------------------------------- classes
# Mặc định: thứ tự alphabet như export Roboflow thường dùng.
# Nhưng ĐỪNG tin mặc định — gọi configure_classes() với `names` đọc từ
# data.yaml gốc để chắc chắn khớp. prepare_data.py làm việc này tự động.
CLASSES = [
    "boots",      # 0
    "gloves",     # 1
    "helmet",     # 2
    "no_boots",   # 3
    "no_gloves",  # 4
    "no_helmet",  # 5
    "no_vest",    # 6
    "person",     # 7
    "vest",       # 8
]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
PERSON_ID = CLS2ID["person"]

# 4 nhóm PPE — mỗi nhóm gộp 1 lớp "có" + 1 lớp "không"
# y_range: vị trí hợp lệ của tâm box PPE, tính theo tỉ lệ chiều cao box người
#          (0.0 = đỉnh đầu, 1.0 = gót chân)
PPE_GROUPS = {
    "helmet": {"pos": "helmet", "neg": "no_helmet", "y_range": (0.00, 0.35)},
    "vest":   {"pos": "vest",   "neg": "no_vest",   "y_range": (0.10, 0.75)},
    "gloves": {"pos": "gloves", "neg": "no_gloves", "y_range": (0.25, 0.85)},
    "boots":  {"pos": "boots",  "neg": "no_boots",  "y_range": (0.60, 1.00)},
}
GROUP_NAMES = list(PPE_GROUPS)          # ['helmet','vest','gloves','boots']
N_GROUPS = len(GROUP_NAMES)

def _rebuild_lookup():
    """Dựng lại bảng tra id -> (nhóm, có phải vi phạm) sau khi đổi CLASSES."""
    _PPE_LOOKUP.clear()
    for g, spec in PPE_GROUPS.items():
        for key, is_violation in (("pos", False), ("neg", True)):
            name = spec[key]
            if name in CLS2ID:
                _PPE_LOOKUP[CLS2ID[name]] = (g, is_violation)


_PPE_LOOKUP = {}
_rebuild_lookup()


def read_names_from_yaml(path):
    """
    Đọc `names` từ data.yaml. Hỗ trợ cả hai dạng Ultralytics dùng:
        names: [a, b, c]            (list)
        names: {0: a, 1: b}         (dict theo index)
    Không cần pyyaml — parse thủ công để tránh thêm phụ thuộc.
    """
    text = Path(path).read_text(encoding="utf-8")

    try:
        import yaml
        data = yaml.safe_load(text)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    except ImportError:
        pass

    # fallback: bắt dạng list một dòng  names: ['a', 'b', ...]
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:") and "[" in s:
            inner = s[s.index("[") + 1: s.rindex("]")]
            return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]

    # fallback: dạng nhiều dòng
    out, collecting = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:"):
            collecting = True
            continue
        if collecting:
            if s.startswith("-"):
                out.append(s[1:].strip().strip("'\""))
            elif ":" in s and not s.startswith("#"):
                key, _, val = s.partition(":")
                if key.strip().isdigit():
                    out.append(val.strip().strip("'\""))
                else:
                    break
            elif s and not s.startswith("#"):
                break
    if out:
        return out
    raise ValueError(f"Không đọc được `names` từ {path}")


def configure_classes(names):
    """
    Nạp thứ tự lớp thật từ dataset. Gọi TRƯỚC mọi thao tác gán nhãn.
    Trả về danh sách cảnh báo (rỗng nghĩa là mọi thứ khớp).
    """
    global CLASSES, PERSON_ID
    CLASSES = list(names)
    CLS2ID.clear()
    CLS2ID.update({c: i for i, c in enumerate(CLASSES)})

    warnings = []
    if "person" not in CLS2ID:
        raise ValueError(
            f"data.yaml không có lớp 'person'. Các lớp tìm thấy: {CLASSES}"
        )
    PERSON_ID = CLS2ID["person"]

    for g, spec in PPE_GROUPS.items():
        for key in ("pos", "neg"):
            if spec[key] not in CLS2ID:
                warnings.append(
                    f"thiếu lớp '{spec[key]}' -> nhóm '{g}' sẽ luôn là không xác định"
                )
    _rebuild_lookup()
    return warnings

# ---------------------------------------------------------------- tham số
IOA_THRESH      = 0.60   # tỉ lệ box PPE nằm trong box người thì mới được gán
CROP_MARGIN     = 0.12   # nới rộng box người 12% mỗi chiều khi cắt
MIN_CROP_SIDE   = 48     # bỏ crop có cạnh ngắn < 48px (người quá xa)
OVERLAP_IOU     = 0.25   # ngưỡng coi là "có người khác lọt vào crop"
MIXED_IS_VIOLATION = True  # đeo 1 chiếc găng -> vẫn tính vi phạm

CROP_W, CROP_H  = 192, 384  # kích thước ảnh đầu vào pha 2 (giữ tỉ lệ người ~1:2)


# ---------------------------------------------------------------- hình học
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO chuẩn hoá -> toạ độ pixel xyxy."""
    return (
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        (cx + w / 2) * img_w,
        (cy + h / 2) * img_h,
    )


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def ioa(inner, outer):
    """Tỉ lệ diện tích `inner` nằm trong `outer`. Khác IoU: không phạt outer to."""
    a = box_area(inner)
    return intersect_area(inner, outer) / a if a > 0 else 0.0


def iou(a, b):
    inter = intersect_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(b, margin, img_w, img_h):
    """Nới box ra `margin` mỗi chiều rồi cắt về trong ảnh."""
    w, h = b[2] - b[0], b[3] - b[1]
    dx, dy = w * margin, h * margin
    return (
        max(0.0, b[0] - dx),
        max(0.0, b[1] - dy),
        min(float(img_w), b[2] + dx),
        min(float(img_h), b[3] + dy),
    )


# ---------------------------------------------------------------- gán nhãn
def spatial_ok(ppe_box, person_box, group):
    """Tâm box PPE có rơi vào vùng hợp lệ trên thân người không."""
    ph = person_box[3] - person_box[1]
    if ph <= 0:
        return False
    rel_y = ((ppe_box[1] + ppe_box[3]) / 2 - person_box[1]) / ph
    lo, hi = PPE_GROUPS[group]["y_range"]
    return lo <= rel_y <= hi


def assign_ppe_to_persons(persons, ppes, use_spatial_prior=True):
    """
    persons : list[xyxy]
    ppes    : list[(cls_id, xyxy)]
    return  : list[dict] cùng độ dài `persons`, mỗi dict là
              {group: {'pos': n, 'neg': n}}

    Một box PPE chỉ được gán cho ĐÚNG MỘT người — người có IoA cao nhất
    trong số những người thoả cả IoA lẫn tiên nghiệm không gian.
    """
    tally = [{g: {"pos": 0, "neg": 0} for g in GROUP_NAMES} for _ in persons]

    for cls_id, pbox in ppes:
        if cls_id not in _PPE_LOOKUP:
            continue
        group, is_violation = _PPE_LOOKUP[cls_id]

        best_i, best_score = -1, 0.0
        fallback_i, fallback_score = -1, 0.0
        for i, person in enumerate(persons):
            score = ioa(pbox, person)
            if score < IOA_THRESH:
                continue
            if use_spatial_prior and not spatial_ok(pbox, person, group):
                if score > fallback_score:      # nhớ lại phòng khi không ai hợp lệ
                    fallback_i, fallback_score = i, score
                continue
            if score > best_score:
                best_i, best_score = i, score

        if best_i < 0:
            best_i = fallback_i          # -1 nếu box PPE không thuộc về ai
        if best_i < 0:
            continue

        tally[best_i][group]["neg" if is_violation else "pos"] += 1

    return tally


def tally_to_labels(tally_one_person):
    """{group:{'pos','neg'}} -> list 4 nhãn theo thứ tự GROUP_NAMES."""
    out = []
    for g in GROUP_NAMES:
        pos, neg = tally_one_person[g]["pos"], tally_one_person[g]["neg"]
        if neg > 0 and pos > 0:
            out.append(1 if MIXED_IS_VIOLATION else -1)   # trạng thái hỗn hợp
        elif neg > 0:
            out.append(1)
        elif pos > 0:
            out.append(0)
        else:
            out.append(-1)                                 # không xác định
    return out


# ---------------------------------------------------------------- đọc nhãn
def read_yolo_label(path, img_w, img_h):
    """Đọc 1 file .txt -> (persons, ppes)."""
    persons, ppes = [], []
    p = Path(path)
    if not p.exists():
        return persons, ppes
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        box = yolo_to_xyxy(*map(float, parts[1:5]), img_w, img_h)
        if cid == PERSON_ID:
            persons.append(box)
        else:
            ppes.append((cid, box))
    return persons, ppes
