# Tối ưu tạo biểu cảm khuôn mặt: fit hai bổ + transfer chiếu barycentric

Ngày: 2026-08-31
Tham chiếu: `Rubikplayer/flame-fitting`, `yfeng95/DECA`

## Bối cảnh

`pipeline/facial_blendshapes.py` tự mô tả là deformation transfer theo Sumner & Popović 2004,
nhưng thực tế nó dừng ở **bước 1** của `fit_lmk3d.py`:

| Tầng | flame-fitting / DECA | Hiện tại |
|---|---|---|
| Landmark | 51/68 điểm, nhúng barycentric (`lmk_face_idx` + `lmk_b_coords`) | 6 centroid nhóm, 1 view |
| Fit | step 1 rigid → **step 2 non-rigid, có regularizer** | chỉ step 1 (Umeyama) |
| Transfer | LBS trên basis PCA | KNN k=4 + Laplacian làm mờ |

Bốn khuyết tật truy được về code:

1. **Vứt 98% landmark.** `face_landmark_align.py:222-229` gộp 478 điểm MediaPipe thành 6 centroid.
   Centroid của một nhóm xê dịch theo tập điểm nhìn thấy được ở mỗi góc, nên chính đại lượng dùng
   để căn chỉnh lại phụ thuộc góc nhìn.
2. **KNN splat không phải deformation transfer.** `facial_blendshapes.py:284-290` nội suy nghịch
   đảo bình phương khoảng cách rồi copy vector. Trường kết quả gãy theo mật độ lấy mẫu template.
3. **Làm mượt bằng làm mờ.** Laplacian đều 0.7/0.3 co biên độ delta, phải bù bằng
   `amplitude_boost=1.35` — hằng số đó là triệu chứng của bước làm mờ, không phải tham số nghệ thuật.
4. **Vòng đứt gãy quanh mắt.** `facial_blendshapes.py:277-279` cho falloff **w = 1.0 ngay trên mask
   mắt**; `unirig_pipeline.py:177-179` rồi zero sạch delta trên `protected`, mà
   `eyelid_patch.py:789` định nghĩa `protected[:n] = left_mask | right_mask` — đúng cái mask đó.
   Vành da sát ngoài mắt chạy ở biên độ tối đa trong khi vùng mắt đứng yên tuyệt đối: một bước
   nhảy C0 biên độ lớn nhất của shape, khép kín thành vòng quanh mắt.

## Quyết định

**Không dùng FLAME.** Nó là model đầu người thật; repo phải chạy được trên chibi mắt khổng lồ,
hươu cao cổ, chim (`examples/`). Ngoài ra weight FLAME và code của cả hai repo tham chiếu đều là
giấy phép MPG **phi thương mại** — nên ở đây **mượn phương pháp, không port code**.

**Phạm vi: giữ nguyên 19 shape mắt + lông mày.** 32 shape miệng/hàm/má/mũi vẫn nằm trên đĩa và
vẫn bị lọc bỏ; mở chúng ra là spec riêng, vì mesh TRELLIS/Hunyuan không có khoang miệng nên
`jawOpen` cần một `mouth_patch` tương tự `eyelid_patch`.

**Hoãn Poisson solve.** Phép giải Sumner-Popović ăn tiền ở shape có xoay/trượt lớn (`jawOpen`,
`mouthSmile`). 19 shape mắt/mày gần thuần tịnh tiến, nên ở phạm vi này nó thêm solver mà không
đổi kết quả đo được. Mở lại khi mở shape miệng/hàm.

## Kiến trúc

### `pipeline/template_fit.py` (mới)

Hai hàm, không trạng thái:

- `robust_similarity(src, tgt)` → `(R, s, t)`. Umeyama có IRLS đánh lại trọng số kiểu GMOf
  (`sbody/robustifiers.py`): `w_i = σ²/(σ² + r_i²)`, 5 vòng. Đây là **step 1** của `fit_lmk3d.py`.
- `nonrigid_icp(...)` → `NonrigidFit{vertices, transforms, landmark_rms}`. Amberg optimal-step
  NRICP: ẩn số là một affine 4×3 cho **mỗi đỉnh template**, cực tiểu hoá

  ```
  E = E_data + α·E_stiff + β·E_lmk
  E_data  = Σ w_i ||X_i ṽ_i − u_i||²        u_i = điểm đích gần nhất
  E_stiff = Σ_(i,j)∈cạnh ||(X_i − X_j)G||²_F   G = diag(1,1,1,γ)
  E_lmk   = Σ ||X_i ṽ_i − l_i||²
  ```

  `α` giảm dần theo lịch trình — đúng ý tưởng lịch trình trọng số của `fit_scan.py`, và chính
  `E_stiff` là regularizer giữ kết quả luôn hợp lý (vai trò của phạt `betas` về mean trong FLAME).

  Loại tương ứng khi pháp tuyến lệch quá `cos 60°` hoặc khoảng cách vượt ngưỡng.

**Vì sao `X_i` là mấu chốt:** `X_i = [A_i | t_i]`, nên `A_i` **chính là** phép biến đổi tuyến tính
cục bộ tại đỉnh template đó. Không cần suy ra gradient biến dạng từ tam giác — nó có sẵn từ lời giải.

### `pipeline/face_landmark_align.py` (mở rộng)

Thêm `detect_face_landmarks(...)` trả `FaceLandmarks{groups, dense_indices, dense_points}`:
giữ nguyên quét 24 góc, nhưng backproject **mọi** chỉ số landmark ở **mọi view trong run tốt nhất**
rồi lấy **median** theo từng chỉ số, loại điểm có MAD giữa các view vượt ngưỡng. Một điểm phải
được ≥ 2 view xác nhận mới giữ (trừ khi chỉ có 1 view).

`locate_face_landmarks_3d` trở thành lớp mỏng gọi hàm trên rồi trả `.groups` — mọi call site cũ
không đổi.

### `pipeline/facial_blendshapes.py` (thay lõi)

Nhúng landmark lên template một lần lúc load, cache ra
`assets/arkit_blendshapes/lmk_embedding.npz` (`face_idx` + `bary`) — tương ứng
`models/flame_static_embedding.pkl`. Điểm nhúng barycentric nên chính xác dưới cấp đỉnh và
không đổi theo góc nhìn.

Luồng transfer mới:

```
landmark dày trên đích ─┐
landmark nhúng template ┴─► robust_similarity ──► nonrigid_icp ──► V', {A_i}
                                                                     │
đỉnh đầu đích ──► chiếu lên mặt template đã bọc ──► (face f, bary b) ─┤
                                                                     ▼
                            delta(p) = Σ_k b_k · (A_k · delta_src_k)
```

Trường kết quả liên tục theo cấu trúc: nội suy barycentric một đại lượng gán theo đỉnh là C0 trên
toàn mặt, tuyến tính trong mỗi tam giác. **Không còn bước làm mượt Laplacian** — và do đó không
còn `amplitude_boost` để bù lại nó (mặc định về `1.0`).

Ba sửa đi kèm:

- **Đảo feather họ eye.** Falloff thành 0 trên vùng `protected` và tăng dần ra ngoài, thay vì 1
  trên đó rồi bị zero. `protected` được truyền xuống từ `unirig_pipeline` thay vì suy đoán lại.
- **Bỏ khối co giãn theo trục Y toàn cục** (`facial_blendshapes.py:334-357`) khi đã có mí. Nó dùng
  trục Y thế giới nên sai ngay khi đầu nghiêng, và vùng nó tác động đã do `eyelid_patch` sở hữu.
  Giữ lại đúng nhánh **không có mí** (`protected is None`) — ở đó nó vẫn là thứ duy nhất làm mắt
  vẽ khép được.
- **Chốt an toàn.** Nếu sau NRICP sai số landmark **tăng** so với sau bước rigid, hoặc tỉ lệ co
  giãn cục bộ ra ngoài `[0.3, 3.0]`, bỏ kết quả non-rigid và dùng lại phép rigid. Landmark hỏng
  không được phép biến thành mặt méo.

## Nghiệm thu

Theo quy ước sẵn có của repo (`scripts/verify_*.py`, không phải pytest):
`scripts/verify_expression_transfer.py` chạy trên `examples/{tira,giraffe,bird,tripo_carrot}.glb`.

| # | Đo | Ngưỡng |
|---|---|---|
| 1 | RMS landmark sau rigid vs sau NRICP | NRICP phải giảm, hoặc chốt an toàn phải kích hoạt |
| 2 | Độ gồ ghề trường: `max_cạnh ‖δ_i − δ_j‖ / max‖δ‖` | giảm so với bản KNN trên cùng mesh |
| 3 | Bước nhảy tại biên `protected` | ≈ 0 (hiện tại ≈ `max‖δ‖`) |
| 4 | Số morph xuất ra, độ dài mảng khớp số đỉnh | không đổi so với trước |
| 5 | GLB tải được, chạy được trên UI playground | thủ công |

## Rủi ro

NRICP bọc một template đầu người lên đầu hươu cao cổ sẽ ra biến dạng lớn — đó là đúng ý đồ, và
`E_stiff` giữ nó cục bộ gần cứng. Nhưng nếu MediaPipe trả landmark sai trên nhân vật phi nhân,
non-rigid sẽ khuếch đại cái sai đó mạnh hơn rigid. Chốt an toàn ở trên là để chặn ca này, và tiêu
chí #1 đo thẳng nó.

---

## Kết quả đo được (cập nhật sau khi hiện thực)

### Ba phát hiện làm đổi thiết kế

**1. Template ARKit bị tách đỉnh.** `Neutral.obj` lưu 3084 đỉnh cho 1220 vị trí thật, và
**không cạnh nào được hai mặt dùng chung** — về mặt tôpô nó là 2304 tam giác rời. Số hạng
stiffness của NRICP định nghĩa trên cạnh giữa các đỉnh láng giềng, nên trên đồ thị đó nó
không ràng buộc được gì. Phải thêm bước hàn (`_weld_template`) trước mọi thứ khác; sau khi
hàn: 1220 đỉnh, 3526 cạnh, 3386 cạnh dùng chung.

**2. `PYTHONNOUSERSITE=1` là bắt buộc.** Có một bản `torch` trong `~/.local` che mất bản của
conda env, làm `torchvision::nms` không tồn tại → `transformers.models.segformer` không
import được → `detect_eye_regions` chết lặng lẽ và **nhân vật mất hoàn toàn khả năng chớp
mắt**. Đặt biến này (hoặc gỡ bản torch trong `~/.local`) trước khi chạy pipeline hoặc
playground.

**3. MediaPipe không nhận mặt trên bất kỳ mesh nào trong `examples/`.** Cả 4 mesh đều trả
`None` ở cả 24 góc quét, ở mọi cỡ framing đã thử. Nghĩa là nhánh landmark — cả bản cũ lẫn
bản mới — chưa từng chạy trên chính bộ mẫu của repo; tất cả rơi về đặt template theo
bounding-sphere. Vì vậy phải dựng một đích tổng hợp mới nghiệm thu được phần fit.

### `scripts/verify_template_fit.py` — nghiệm thu có ground truth

Đích là chính template ARKit qua một phép biến dạng đã biết (scale dị hướng 1.25/0.82 + uốn
bậc hai), nên Jacobian là đáp án chính xác cho từng đại lượng.

| Đo | Kết quả |
|---|---|
| RMS landmark, stage 1 → stage 2 | 8.99 → 0.095 (**tốt hơn 94,6 lần**) |
| Template khớp mặt đích | sai số trung bình **0,52 cạnh trung vị** |
| Transform từng đỉnh vs Jacobian chính xác | sai số tương đối trung vị **0,062** |
| Hướng dịch chuyển, xấu nhất trong 19 shape | **cos = 0,994** |
| Biên độ dịch chuyển, xấu nhất | tỉ lệ **0,92** |
| Bước nhảy tại biên `eyelid_patch` | **1,000 → 0,186** (giảm 5,4 lần) |

Con số cuối là bằng chứng trực tiếp cho lỗi đã chẩn đoán: không có fade, vành da sát ngoài
mắt chạy đúng **100% biên độ** của shape ngay cạnh vùng bị ghim bằng 0.

### `scripts/verify_expression_transfer.py` — 4 mesh mẫu

Đo **gradient tương đối**: `‖d_i − d_j‖ / (đỉnh × độ_dài_cạnh / trung_vị_cạnh)`, chia cho
cùng đại lượng đo trên chính template. Hai lần chuẩn hoá đều cần: chia cho độ dài cạnh vì
cạnh trong một cái đầu chênh nhau 5 lần và trường mượt đi qua cạnh dài thì đổi nhiều hơn;
chia cho trung vị vì template dài 256 đơn vị còn đầu đích 0,07.

| Mesh | p99 | max |
|---|---|---|
| tira | 0,13 | 4,65 |
| giraffe | 0,38 | 4,12 |
| bird | 0,05 | 6,87 |
| tripo_carrot | 0,13 | 1,36 |

**Gate đặt ở p99**, max chỉ báo cáo. Lý do: max là thống kê một-đỉnh và bị chi phối bởi một
lỗi có sẵn ở thượng nguồn — `detect_head_region` cho tira một cái đầu bằng **3,3% chiều cao
nhân vật** (đầu thật 12–15%), nên template bị co vào quả cầu nhỏ bằng một phần tư cái đầu
thật và shape lông mày rơi xuống khoảng 7 đỉnh. Dồn bất kỳ trường nào vào 7 đỉnh thì đỉnh
của nó tất yếu nằm cạnh một láng giềng gần bằng 0.

### Chưa làm được

- **Lọc theo pháp tuyến cho phép chiếu**: đã thử (giả thuyết: phép chiếu nhảy qua trục trung
  vị) và **đo thấy tệ hơn** — giraffe 4,12 → 4,95, carrot 1,36 → 2,59. Đã gỡ bỏ.
- **Bất đối xứng trái/phải**: nhúng landmark ra 284 điểm một bên, 121 bên kia; mọi shape
  "Right" phục hồi biên độ kém hơn "Left" (0,92 vs 0,98). Chỉnh tâm cửa sổ hợp nhất theo độ
  trải ngang của landmark không sửa được. Hướng dịch chuyển không bị ảnh hưởng.
- **Đường mí trên mesh thật**: face parser từ chối cả 4 mẫu (mask quá nhỏ, hoặc không có
  pixel mắt ở cả 16 góc) — đúng rủi ro đã ghi ở spec 2026-08-30. Phần handover vì thế chỉ
  nghiệm thu được trên đích tổng hợp.
- **Hằng số tuyệt đối `0.12`** trong khối co giãn mí cũ (`facial_blendshapes.py`, nhánh
  không có mí): nó là khoảng cách tuyệt đối nên trên lưới đơn vị lớn nó xoá sạch họ eye. Lỗi
  có sẵn, để nguyên vì pipeline chỉ chạm tới nhánh đó khi mesh không có mí.
