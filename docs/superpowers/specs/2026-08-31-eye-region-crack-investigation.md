# Vết nứt vùng mắt: điều tra nguyên nhân gốc

Ngày: 2026-08-31
Ca tái hiện: `playground/storage/job_1788109378_3d_style_3/stage1_prep/*_input.glb`
(268.225 đỉnh, texture 4096², nhân vật kiểu Mario)

## Triệu chứng

Ở `eyeBlink = 1` mắt **không khép**; giữa mỗi mắt là một mảng xám răng cưa. Ảnh:
`test_results/eye_debug/` và `test_results/eye_closure/`.

## Dụng cụ đo mới

`scripts/verify_eye_closure.py` — render thật, đếm phần con mắt còn lộ ra sau khi chớp.

**Phiên bản đầu của chính nó bị sai và đã phải sửa hai lần**, ghi lại vì cái bẫy này tốn
nhiều thời gian nhất:

1. Render có đèn → da cháy sáng tới luma trung vị 0,927, lòng trắng lẫn với má. Phải render
   **phẳng, không đèn** để pixel bằng albedo.
2. Đo trên một dải ngang khuôn mặt → chọn trúng **tóc, lông mày, bộ ria**: 55.802 pixel
   "mắt" mà **không một pixel nào là lòng trắng**. Những thứ đó không đổi khi chớp, nên
   điểm số đứng im ở 99% bất kể mí làm gì. Phải **chiếu đỉnh của từng mắt** rồi chỉ đo
   trong khung của nó.

Sau khi sửa: 16.757 pixel mắt, closure **97,5%** — khớp với ảnh.

## Chuỗi nhân quả đã xác lập

**1. `detect_head_region` đói độ phân giải cho parser.** Nó trả quả cầu bán kính bằng
**45,6% chiều cao nhân vật** (và với `examples/tira.glb` là 3,3% — sai cả hai chiều). Khung
render vì thế chứa cả thân: histogram nhãn một view đọc `cloth=106473, skin=44540`, hai mắt
cộng lại chỉ 145×33 px.

| Bán kính đóng khung | Pixel mắt / view | Đỉnh nâng được |
|---|---|---|
| 0,4566 (mặc định) | 938 / 1298 / 886 | 98 / 171 / 100 |
| 0,1142 | 32857 / 47041 / 20097 | 583 / 1138 / 205 |

**2. Mask là *mẫu*, không phải *vùng*.** `_lift_blob` trả một đỉnh cho mỗi **pixel** được
gán nhãn, nên kích thước mask bị chặn bởi số pixel, không liên quan mật độ lưới. Mask thu
được 76/128 đỉnh trong khi con mắt trải trên **726 đỉnh / 943 tam giác**.

**3. Mí sao chép đúng các mặt của mask.** `_build_lid`: `sel = faces[inside >= 1]`, và
`region ⊂ mask`. Nên **độ phủ của mí = độ phủ của mask = 21,3% diện tích nhìn thấy của
mắt**. Đó chính là mảng răng cưa giữa mắt.

**4. Màu mí sai theo.** `_skin_colors` lấy màu từ vành hai bước ngoài mask; mask thưa nên
vành đó rơi *vào trong lòng mắt* → mí ra màu xám thay vì màu da.

**Cơ chế mí thì đúng**: đo ở weight 1, đỉnh mí nằm cách mặt lưới gốc **0,4 cạnh trung vị** —
nó bám đúng bề mặt, chỉ là bám trên 21% diện tích.

## Bốn bản vá đã thử và thất bại (đã revert hết)

| Thử | Kết quả |
|---|---|
| Nở mask theo luma so với da | 61/120 → 353/1000 nhưng **rò ra thái dương và lông mày**; closure 95,4% |
| Lift theo chiều thuận (chiếu đỉnh vào pixel) | 52/163 — gần như y cũ |
| Đóng khung lại quanh hai mắt rồi parse lần hai | 278/105, **rò một mảng lên trán** |
| Nở theo texture, láng giềng gần nhất trong RGB, thang đo neo theo khoảng cách hai mắt | 76/129 → 117/149; closure không đổi |

## Vì sao texture không cứu được ca này

Đo trên chính nhân vật: **albedo lòng trắng 0,84, albedo da 0,785 — chênh 0,055.** Cảm giác
"mắt trắng" đến từ tròng đen và viền tối, không từ độ sáng của lòng trắng. Trong 917 đỉnh
quanh mắt chỉ có **39 đỉnh** sáng hơn 0,85, và cả 39 đã nằm sẵn trong mask.

Không phải texture vô dụng nói chung — nhưng trên nhân vật này nó **gần như không mang tín
hiệu** để tách mắt khỏi da. Tín hiệu mạnh lại nằm ở **hình học**: hốc mắt lõm sâu bằng
**0,54 và 1,42 lần bán kính mắt** (đo dọc pháp tuyến mặt). Đó là thứ nên dùng, và là thứ
chưa bản vá nào của tôi động tới.

## Đã sửa và giữ lại

- **Đóng khung vòng quét vào khuôn mặt** (`_face_ball`): dùng chính parser tìm mặt bằng các
  nhãn da/mũi/miệng — chúng phủ hàng nghìn pixel nơi con mắt chỉ có hàng trăm — rồi đóng
  khung lại. Không đụng `detect_head_region` nên skeleton và segmentation không đổi. Đo
  được: bán kính 0,4566 → 0,2476, parse confidence 0,704 → 0,810, 11 blob trên 8 view
  (trước: 10 trên 7).
- **Chốt an toàn của bước non-rigid bị so sai đại lượng** (lỗi của phiên trước, trong
  `facial_blendshapes._fit_template`): `fit.transforms` tác dụng lên template *đã* căn
  chỉnh nên phải so với ma trận đơn vị, nhưng lại bị so với phép rigid vốn mang đơn vị của
  template (~256 so với 0,17 của đích). Tỉ số đọc ra 366 và **bước non-rigid bị vứt bỏ trên
  mọi nhân vật thật**. Sau khi sửa, trên nhân vật này: RMS landmark **0,0265 → 0,0111**.

## Việc tiếp theo, theo thứ tự

1. **Dựng mí từ hình học hốc mắt, không từ mask đỉnh.** Khớp một ellipse trên mặt từ tâm +
   kích thước + hướng (những đại lượng đã đo thấy ổn định qua mọi lần chạy), rồi sinh nắp mí
   ở độ phân giải của chính nó. Mask khi đó chỉ cần nói *mắt ở đâu*, không cần nói *đỉnh nào
   thuộc mắt* — mà đó đúng là thứ parser làm tốt và thứ nó làm dở.
2. **Sửa `detect_head_region`.** Nó sai cả hai chiều và là nguồn đói độ phân giải; `_face_ball`
   mới chỉ che cho vòng quét mắt.
