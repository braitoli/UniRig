# UniRig Handoff — Complete Skeleton + Skin + Animation Pipeline & Web Playground

> **Trạng thái:** ✅ **HOÀN THÀNH TOÀN DIỆN** (Full End-to-End Pipeline & Web Playground LAN)
> - **Input 3D:** `.glb`, `.gltf`, `.obj`
> - **Stage 1 (Preprocess):** Chuẩn hóa mesh bằng `trimesh`
> - **Stage 2 (Skeleton):** UniRig AR 350M transformer (`sdpa` attention)
> - **Stage 3 (Skin Weights):** Bone-segment proximity weighting kết hợp surface Laplacian graph diffusion
> - **Stage 4 (Rigged GLB & Animation):** Xuất file standard glTF 2.0 / GLB có skins, joints, IBM, weights và 5 motion clip (`Idle`, `Walk`, `Run`, `Wave`, `Dance`)
> - **Web Playground (Three.js):** Giao diện tương tác 3D đa giai đoạn, hiển thị heatmap da theo từng khớp xương, mixer điều khiển animation, lưu trạng thái tự động resume khi F5, xem lịch sử và publish ra mạng LAN.

---

## 1. Truy cập Web Playground (LAN & Local)

Web Playground đang chạy nền trên port `7860`:

- **LAN URL:** `http://192.168.1.43:7860`
- **Local URL:** `http://localhost:7860`

### Tính năng Web Playground:
1. **4 Tab Preview 3D (Three.js):**
   - **Tab 1 (Mesh Thô):** Xem mô hình input, hỗ trợ wireframe, shading, xoay/phóng to/thu nhỏ.
   - **Tab 2 (Skeleton):** Hiển thị overlay hệ xương (các khớp hình cầu cyan và đoạn xương cylinder đỏ) cùng cây phân cấp (Bone Hierarchy Tree).
   - **Tab 3 (Skin Heatmap):** Nhấp chọn bất kỳ khớp xương nào trên cây để xem Heatmap màu tương tác (Blue = 0.0 → Cyan → Green → Yellow → Red = 1.0) biểu thị vùng ảnh hưởng của khớp đó trên bề mặt mesh.
   - **Tab 4 (Animation 3D):** Trực tiếp play/pause 5 animation procedural (`Idle`, `Walk`, `Run`, `Wave`, `Dance`), điều chỉnh tốc độ từ `0.25x` đến `2.0x`.
2. **Preset & Upload Tùy Chọn:**
   - Sẵn 4 preset: `Bird`, `Giraffe`, `Tira`, `Carrot`.
   - Upload file kéo thả cho bất kỳ file `.glb` / `.gltf` / `.obj` tùy chỉnh nào.
3. **Save State & Resume khi F5:**
   - Toàn bộ jobs, metadata, tiến trình và kết quả được lưu trữ bền vững vào SQLite (`playground/storage/playground.db`).
   - Khóa phiên (`localStorage`) tự động khôi phục đúng model, stage và góc nhìn 3D khi tải lại trang (F5).
4. **Lịch sử sinh (History):**
   - Xem lại tất cả các model đã xử lý trong quá khứ, click để xem lại 3D ngay lập tức.
   - Nút tải về: Rigged Animated `.glb` và Skeleton `.obj`.

---

## 2. Kết quả kiểm thử End-to-End trên 4 Model

| Model | Vertices | Faces | Số Bones | Thời gian chạy | Kết quả GLB |
|---|---|---|---|---|---|
| `bird.glb` | 4,740 | 9,477 | 26 bones | ~19.5s | `bird_rigged_animated.glb` (551 KB) |
| `giraffe.glb` | 10,820 | 21,636 | 21 bones | ~18.5s | `giraffe_rigged_animated.glb` (926 KB) |
| `tira.glb` | 25,000 | 50,000 | 52 bones | ~61.6s | `tira_rigged_animated.glb` (5.26 MB) |
| `tripo_carrot.glb` | 25,000 | 50,000 | 7 bones | ~22.1s | `tripo_carrot_rigged_animated.glb` (10.7 MB) |

---

## 3. Cấu trúc Source Code thêm mới / tinh chỉnh

```
UniRig/
├── pipeline/
│   ├── rig_export.py          # Pure-Python glTF 2.0/GLB skinned mesh exporter (IBM, skins, animations)
│   ├── skinning.py            # Bone-segment distance & Laplacian surface smoothing skin solver
│   ├── animation.py           # Procedural motion generator (Idle, Walk, Run, Wave, Dance)
│   └── unirig_pipeline.py     # Unified 4-stage pipeline runner
├── playground/
│   ├── server.py              # FastAPI server + background worker + LAN IP detector
│   ├── database.py            # SQLite state persistence & history
│   ├── storage/               # Saved jobs, npz, glb & database
│   └── static/
│       ├── index.html         # Responsive 3D Playground UI
│       ├── styles.css         # Dark theme UI stylesheet
│       └── app.js             # Three.js viewport, heatmap shader, animation mixer, state sync
├── preprocess_trimesh.py      # Mesh preprocessor (no bpy required)
└── run_skeleton.sh            # Skeleton CLI runner
```

---

## 4. Cách khởi động Playground thủ công (khi cần)

```bash
# Kích hoạt môi trường
source /home/braitoli/miniconda/etc/profile.d/conda.sh
conda activate unirig312
export PYTHONNOUSERSITE=1
cd /home/braitoli/workspace/namnh/code/poc/UniRig

# Chạy Playground server
python playground/server.py
```
Mở trình duyệt trên bất kỳ thiết bị nào trong cùng mạng LAN: `http://192.168.1.43:7860`
