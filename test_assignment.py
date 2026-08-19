"""
Smoke test cho logic gán PPE -> người. Chạy: python test_assignment.py
Không cần dataset, không cần GPU.
"""

from common import (
    CLS2ID, GROUP_NAMES, assign_ppe_to_persons, ioa, spatial_ok, tally_to_labels,
)

ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}\n        got  {got}\n        want {want}")


# người cao 200px, rộng 60px, ở (100,50)-(160,250)
P1 = (100.0, 50.0, 160.0, 250.0)
# người thứ hai đứng cạnh, chồng lấn một phần
P2 = (145.0, 50.0, 205.0, 250.0)

print("\n[1] Vị trí trên thân người")
check("mũ ở đỉnh đầu hợp lệ",
      spatial_ok((110, 52, 150, 78), P1, "helmet"), True)
check("mũ ở ngang chân KHÔNG hợp lệ",
      spatial_ok((110, 220, 150, 245), P1, "helmet"), False)
check("giày ở dưới cùng hợp lệ",
      spatial_ok((105, 225, 155, 248), P1, "boots"), True)
check("áo ở giữa thân hợp lệ",
      spatial_ok((104, 100, 156, 170), P1, "vest"), True)

print("\n[2] IoA")
check("box nằm trọn trong người -> 1.0",
      round(ioa((110, 60, 150, 90), P1), 3), 1.0)
check("box nằm ngoài -> 0.0",
      round(ioa((300, 300, 340, 340), P1), 3), 0.0)

print("\n[3] Gán cơ bản — một người, không mũ, có áo")
t = assign_ppe_to_persons([P1], [
    (CLS2ID["no_helmet"], (110, 55, 150, 80)),
    (CLS2ID["vest"],      (104, 105, 156, 175)),
])
check("nhãn suy ra", tally_to_labels(t[0]), [1, 0, -1, -1])
print(f"        thứ tự nhóm: {GROUP_NAMES}")

print("\n[4] Hai người chồng lấn — mũ phải về đúng chủ")
# mũ nằm hẳn trên đầu người 2
helmet_p2 = (170, 55, 200, 80)
t = assign_ppe_to_persons([P1, P2], [(CLS2ID["helmet"], helmet_p2)])
check("người 1 không nhận mũ", tally_to_labels(t[0])[0], -1)
check("người 2 nhận mũ",       tally_to_labels(t[1])[0], 0)

print("\n[5] Tiên nghiệm vị trí cứu trường hợp nhập nhằng")
# giày của người 1, nằm trong vùng chồng lấn theo chiều ngang
boots = (148, 228, 158, 246)
with_prior = assign_ppe_to_persons([P1, P2], [(CLS2ID["boots"], boots)])
check("có tiên nghiệm: cả hai đều thấy hợp lệ về y, người IoA cao hơn thắng",
      sum(1 for x in with_prior if x["boots"]["pos"] > 0), 1)

print("\n[6] Trạng thái hỗn hợp — đeo một chiếc găng")
t = assign_ppe_to_persons([P1], [
    (CLS2ID["gloves"],    (98, 150, 112, 168)),
    (CLS2ID["no_gloves"], (150, 150, 164, 168)),
])
check("tính là vi phạm (MIXED_IS_VIOLATION=True)",
      tally_to_labels(t[0])[GROUP_NAMES.index("gloves")], 1)

print("\n[7] PPE mồ côi — không thuộc về ai")
t = assign_ppe_to_persons([P1], [(CLS2ID["helmet"], (400, 400, 430, 425))])
check("không gán cho người nào", tally_to_labels(t[0]), [-1, -1, -1, -1])

print("\n[8] Người không có nhãn PPE nào -> toàn bộ không xác định")
t = assign_ppe_to_persons([P1], [])
check("4 nhãn đều -1", tally_to_labels(t[0]), [-1, -1, -1, -1])

print(f"\n{'=' * 44}\n  {ok} pass, {fail} fail\n{'=' * 44}")
raise SystemExit(1 if fail else 0)
