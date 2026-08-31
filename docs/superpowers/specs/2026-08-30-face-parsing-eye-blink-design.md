# Face parsing cho nhận diện mắt + hiệu ứng chớp/nheo mắt

Ngày: 2026-08-30

## Bối cảnh

`pipeline/eye_detection.py` hiện dùng OWLv2 (`google/owlv2-base-patch16-ensemble`) với prompt
`"eyeball"`, trả về **bounding box**. Box được backproject qua depth buffer thành một khối vertex
chứa cả má và vành mí. `pipeline/facial_blendshapes.py` phải chẻ đôi khối đó theo trục Y toàn cục
rồi kéo nửa trên xuống — đó là lý do blink hay gập má trên đầu nghiêng.

Repo `braitoli/FaceParsing` chứa hai lõi AI khác nhau:

- `app/eye_detector.py` dùng `deepghs/anime_eye_detection` (YOLO box). **Không dùng**: docstring
  của `pipeline/eye_detection.py` ghi lại rằng model này đã được đo trên ~500 cấu hình ở đây và
  chưa từng định vị đúng một con mắt nào.
- `app/model.py` + `app/eye_mesh_locator.py` dùng `jonathandinu/face-parsing` (SegFormer,
  CelebAMask-HQ 19 nhãn) — trả **mask pixel-level**. Đây là phần được port.

## Quyết định

1. **Thay thế hoàn toàn** OWLv2 bằng SegFormer face parsing. Không giữ đường dự phòng.
2. Phạm vi nhãn: mắt (`l_eye=4`, `r_eye=5`, `eye_g=3`) **và lông mày** (`l_brow=6`, `r_brow=7`).
3. Hiệu ứng chớp mắt dùng **eyelid patch**: mesh sinh ra là một khối liền, mắt là màu vertex vẽ
   trên da (đã đo trên GLB thật: 1 geometry, `body_count=1`, `ColorVisuals`). Không có nhãn cầu,
   không có mí. Nên blink được làm bằng cách sinh thêm một mảng hình học màu da phủ lên mắt.
4. **Có** sinh clip auto-blink, xuất bằng animation channel `path="weights"`.

## Kiến trúc

### Port ở mức nào

Giữ khung của UniRig, chỉ port phần logic quyết định. Không port:

- Software rasterizer thuần numpy của FaceParsing (`_rasterize`/`_Shader`/`_render`, ~220 dòng) —
  tồn tại vì họ chạy trên Mac không EGL. UniRig đã có pyrender+EGL chia sẻ context trong
  `pipeline/head_views.py`.
- `_upright_rotation` / `_head_crop` — trùng với `pipeline/mesh_segmentation.detect_head_region`,
  vốn dùng đúng cùng model ONNX QtMeshEditor.

Port sang:

| Từ FaceParsing | Lý do |
|---|---|
| `jonathandinu/face-parsing` + label id | Mask pixel-level thay cho box |
| Gating theo tỉ lệ phiếu (`_MIN_VOTE_RATIO`) trên số góc *nhìn thấy được* | Phân biệt mắt thật với chi tiết trang trí to |
| `_normal_coherence` (khử winding qua tensor bậc 2) | Chặn nhóm vertex tán loạn |
| `_mirror_plane` + `_mirror_missing_eyes` | Sửa lỗi thật: `_select_pair` bắt buộc tìm được *cặp* nên hươu cao cổ/chim luôn trả `None` |

### Module

- **`pipeline/face_parsing.py`** (mới) — SegFormer lazy singleton, `parse_faces(images)` trả
  `(labels, confidence)` theo batch. Pixel có confidence dưới `MIN_CONFIDENCE=0.60` bị coi là
  chưa phân loại.
- **`pipeline/eye_detection.py`** (viết lại lõi) — giữ nguyên `EyeRegions` và chữ ký
  `detect_eye_regions(...) -> Optional[EyeRegions]`. `_lift_box` → `_lift_blob`;
  `_select_pair` → `_finalize_groups` + `_mirror_missing_eyes`. `EyeRegions` thêm
  `left_brow_mask` / `right_brow_mask`.
- **`pipeline/eyelid_patch.py`** (mới) — sinh mí mắt.
- **`pipeline/rig_export.py`** (mở rộng) — animation channel `path="weights"` nhắm node 0 (node
  mesh), accessor `SCALAR`, `count = n_frames * n_morph_targets`.
- **`pipeline/animation.py`** (thêm hàm) — `generate_blink_animation()`.

### Eyelid patch — cơ chế (đã sửa lại sau khi render kiểm tra)

Mỗi mắt sinh **hai** mí. Ở weight 0 mỗi đỉnh mí bị bóp lên **đường viền mắt tại đúng cột ngang
của nó** — mí thành một vệt mảnh dọc chân mi (2–5% diện tích mắt), tức một nếp mí. Morph trải
nó xuống phủ mắt.

Bốn điểm quyết định, đều rút từ cách rig game/phim dựng mắt:

- **Mí trên đi 75%, mí dưới đi lên gặp nó.** Đường khép nằm thấp, không cắt ngang giữa mắt.
- **Khoé mắt gần như đứng yên.** Không phải hệ số tô tay: nó tự sinh ra vì mỗi đỉnh xuất phát
  từ viền mắt ngay trên nó, mà ở khoé viền mắt khép lại nên không còn gì để đi. Đường phân chia
  hai mí vì thế phải là **đường cong theo chiều cao từng cột**, không phải đường ngang.
- **Chuyển động theo cung, không cắt ngang nhãn cầu.** Khớp cầu vào vùng mắt lấy bán kính, mỗi
  đỉnh được đẩy ra thêm đúng sagitta của cung nó quét.
- **Không nấp dưới da.** Đã thử và sai: morph nội suy tuyến tính nên mọi đỉnh chui khỏi da ở
  cùng một mốc (đo được 0.79) → mắt trông y nguyên suốt 3/4 cái nháy rồi mí bật ra một lúc.

Phủ kín có chứng minh: mí trên nhận **mọi mặt có ít nhất 1 đỉnh trên đường khép**, mí dưới nhận
mặt nằm trọn dưới nó ⇒ không mặt nào của mắt bị cả hai bỏ sót. Quy tắc "đủ 3 đỉnh" cho cả hai
làm thủng một hàng lỗ dọc đường khép.

`eyeSquint` là **AU7**, không phải nháy nửa chừng: mí dưới 0.55, mí trên 0.15, kèm sinh
`cheekSquint*` (AU6 nâng má, siết khoé ngoài). `eyeWide` là **AU5** — chỉ nâng mí trên.

- Màu: mỗi đỉnh mí lấy màu của đỉnh gần nhất trên **vành ngay ngoài mask** (màu da).
- Skin weight: đỉnh mới thừa kế nguyên weight của đỉnh gốc ⇒ không phải chạy lại skinning.

### Nhãn cầu — sửa lỗi con ngươi bị méo

Con ngươi là **màu vẽ**. Mọi morph làm biến dạng bề mặt đều áp một trường dịch chuyển lên nó, mà
trường dịch chuyển biến hình tròn thành hình elip — đó là lý do nhóm `eyeLook*` (lấy từ ARKit
transfer) làm vỡ mống mắt. Phép biến đổi duy nhất giữ hình tròn vẫn tròn là **phép biến đổi cứng**.

Vì thế mống mắt được **tách khỏi mặt thành một chỏm riêng và xoay quanh tâm nhãn cầu đã khớp** —
cấp cho nhân vật cái nhãn cầu nó chưa từng có.

- Tách lòng đen bằng ngưỡng độ sáng trong vùng mắt (`_IRIS_LUMA_SPLIT = 0.55`); mắt cách điệu
  tối toàn bộ thì lấy cả vùng, vẫn đúng.
- Chỏm nổi `0.15 x median_edge` — **ít hơn cả hai mí**, nên mí vẫn khép đè lên nó.
- 4 hướng nhìn = xoay quanh trục `cross(outward, direction)` qua tâm cầu. Góc bị chặn bởi
  **khoảng trống thật của khe mắt** (`(nửa_mắt − nửa_mống) / bán_kính`, trần 24°) nên mống mắt
  không bao giờ trượt ra má.
- Lòng trắng dưới chỏm được sơn lại. An toàn vì ở weight 0 chỏm che đúng vùng đó — ngoại hình
  nhân vật không đổi cho tới khi nó thật sự liếc.
- Nhân vật có texture (`colors=None`) dùng kênh `appearance` riêng: texture nướng xuống per-vertex
  chỉ để *quyết định* chỗ nào là mống mắt, không xuất thành COLOR_0.

### Đóng băng vùng mắt vẽ

`EyelidResult.protected` = vùng mắt vẽ ∪ mọi đỉnh mí ∪ mọi đỉnh chỏm. Mọi delta từ ARKit transfer
bị **zero trên vùng này** trước khi gộp. Nguyên tắc: *không morph nào được dịch chuyển một đỉnh
mang màu của mắt, trừ chỏm mống mắt, và chỏm chỉ xoay.*

Khối 6b cũ trong `facial_blendshapes.py` bị xoá hẳn — mọi shape trong đó đều làm biến dạng chính
vùng mắt vẽ. `eyeBlink`/`eyeSquint`/`eyeWide`/`eyeLook*` giờ đều do `eyelid_patch` cấp.

`eyeWide` (AU5) trên mặt có mắt vẽ: khe mắt không thể to ra được, nên thứ chuyển động là **vành da
quanh mắt** (trên nâng 0.16, dưới hạ 0.08 lần chiều cao mắt), mắt không bị đụng tới.

### Timing auto-blink

Đóng 40 ms, giữ 20 ms, mở 180 ms có ease-out (mở chậm hơn đóng 4,5 lần — sinh lý học cho
30–50 ms đóng và 150–300 ms mở). Hai mắt lệch 15 ms. Khoảng cách giữa các nháy lấy từ phân phối
mũ (trung bình ~4,5 s, sàn 1,5 s), 12% xác suất nháy đúp. Keyframe thưa, đặt đúng chỗ đường cong
gãy, vì glTF nội suy LINEAR.

### Thứ tự bắt buộc

Patch phải được ghép vào mesh **trước** khi sinh morph ARKit, nếu không mọi mảng `(N,3)` lệch
chiều. `detect_eye_regions` chạy **một lần** ở `unirig_pipeline` rồi truyền xuống
`transfer_blendshapes` qua tham số `eye_regions`, thay vì để nó tự dò lại (16 render + SegFormer).

## Luồng dữ liệu

```
mesh (V, F, vertex_colors)
 └─ detect_head_region ────────────────────────► HeadRegion
     └─ HeadViewRenderer (16 yaw + pitch retry) ─► View[] {rgb, depth, pose}
         └─ face_parsing.parse_faces ───────────► labels + confidence
             └─ blob liên thông / view (eye | eye_g | brow)
                 └─ backproject_pixels ─────────► tập vertex 3D / blob
                     └─ gom cụm theo tâm 3D
                         └─ gate tỉ lệ phiếu + normal coherence
                             └─ mirror-plane hoàn thiện cặp
                                 └─ EyeRegions {left, right, brows}
                                     ├─ eyelid_patch.attach_eyelids ─► V', F', colors', weights'
                                     │                                  + morph eyeBlink/eyeSquint
                                     ├─ facial_blendshapes (ARKit, có mày thật)
                                     └─ generate_blink_animation ────► track "weights"
                                                                        └─ rig_export ─► GLB
```

### Mỗi biểu cảm mắt là một clip

Morph target là một **tư thế**, không phải một **màn diễn**. Trả về `eyeSquintLeft = 0.85` chỉ nói
khuôn mặt trông thế nào, không nói nó tới đó bằng cách nào — ứng dụng set thẳng weight sẽ ra cú
nhảy hình. Vì vậy mỗi preset trong `EXPRESSION_PRESETS` được xuất thành clip `weights` riêng, có
tấn công / giữ / nhả.

Động học **không dùng chung một đường bao**, vì đó chính là thứ phân biệt các biểu cảm:

| Nhóm | Tấn công | Giữ | Nhả | Vì sao |
|---|---|---|---|---|
| Liếc / đảo mắt | 70 ms | 1,10 s | 90 ms | Saccade là chuyển động đạn đạo 30–100 ms, thuộc loại nhanh nhất cơ thể làm được. Đưa vào trong 1/4 giây sẽ ra "quay đầu", không phải "liếc" |
| Nháy một mắt | 60 ms | 200 ms | 140 ms | Như một cái nháy có chủ ý |
| Nhắm 2 mắt | 50 ms | 100 ms | 180 ms | Giữ bất đối xứng đóng-nhanh/mở-chậm |
| Mở to (giật mình) | 90 ms | 700 ms | 420 ms | Vào gần nhanh như saccade, ra chậm |
| Nheo / nhíu | 200 ms | 900 ms | 360 ms | Cơ chủ động, chậm đều hai đầu |

Tấn công ease-out, nhả ease-in, đều lấy mẫu bằng keyframe vì glTF nội suy LINEAR. Sau khi nhả có
0,6 s nghỉ để clip lặp không dồn dập.

Preset mà mesh **không lái được morph nào** thì không sinh clip — một mục rỗng trong danh sách
animation còn tệ hơn là không có. Preset lái được một phần vẫn ra clip cho phần đó.

Phía UI: nút biểu cảm phát clip nếu có, ngược lại rơi về set weight tĩnh như cũ. Tên clip lấy từ
`EXPRESSION_PRESETS[...]["name"]` — định nghĩa một chỗ duy nhất trong `facial_blendshapes.py`, client
đọc qua `/api/facial_blendshapes/presets` nên nút, clip xuất ra và dropdown không lệch nhau.

## Nghiệm thu

| # | Kiểm tra |
|---|---|
| 1 | `scripts/debug_head_eye.py` dump mask SegFormer từng view trên chibi / giraffe / bird |
| 2 | giraffe & bird ra đủ 2 mắt (hiện `_select_pair` trả `None`) |
| 3 | Render mesh có patch ở weight=0 vs mesh gốc — pixel diff xấp xỉ 0 |
| 4 | Ở weight=1 vùng mắt không còn pixel màu con ngươi |
| 5 | GLB load được bằng Three.js r128, mắt tự nháy |

## Rủi ro đã biết

SegFormer được train trên ảnh chân dung người thật (CelebAMask-HQ). FaceParsing báo cáo nó bắt
được mắt thú bốn chân ở confidence 0.79-0.96 khi framing đúng, nhưng đó là mesh test của họ.
Nếu nó im lặng trên nhân vật do TRELLIS/Hunyuan3D sinh ra, `detect_eye_regions` trả `None` và
nhân vật mất hẳn khả năng chớp mắt — OWLv2 đã bị gỡ. Người dùng đã chọn chấp nhận rủi ro này và
tự chạy thử trên UI để quyết định.
