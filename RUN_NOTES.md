# UniRig trên máy aarch64 (GB10 / NVIDIA Blackwell) — ghi chú chạy

Project gốc ở `https://github.com/VAST-AI-Research/UniRig` được thiết kế cho
Linux x86_64 + Python 3.11 + CUDA 12.x. Trên máy bạn là
**aarch64 (ARM64) + GB10 + CUDA 13.0 + Python 3.12**, nên nhiều dependency
chính không có wheel sẵn. Phần này tóm tắt những gì đã chỉnh và cách chạy.

## Kết quả

Đã chạy **skeleton prediction** thành công cho cả 4 example:

| Input | Output `.obj` (số bone = `faces / 2`) | Thời gian |
| --- | --- | --- |
| `examples/giraffe.glb` | 82 lines · 60 verts · 20 bones | ~5s |
| `examples/bird.glb`    | 102 lines · 75 verts · 25 bones | ~6s |
| `examples/tira.glb`    | 206 lines · 153 verts · 51 bones | ~12s |
| `examples/tripo_carrot.glb` | 26 lines · 18 verts · 6 bones | ~3s |

Model: `experiments/skeleton/articulation-xl_quantization_256/model.ckpt`
được tự động tải về từ HuggingFace (`VAST-AI/UniRig`).

## Môi trường

- Conda env: `unirig312` (Python 3.12 — khớp với Python mà Blender dùng, cần
  thiết cho bất cứ chỗ nào muốn dùng `bpy`).
- GPU: `NVIDIA GB10`, CUDA 13.0, driver 580.159.03. PyTorch
  `2.13.0+cu130` (sm_120 / Blackwell).
- Quan trọng: trong `~.local/lib/python3.12/site-packages` có bản
  `torch-2.13.0+cpu` rỗng đè lên env. Mọi script đều phải chạy với
  `PYTHONNOUSERSITE=1` để không bị shadow, nếu không `torchvision::nms`
  sẽ văng lỗi dispatch lúc import.
- Cũng cần `nvidia-cusolver` (>=12.0) được cài **bên trong** env — thiếu
  cái này torch load được nhưng bất kỳ call CUDA nào cũng `SIGBUS` ngay
  khi load `libcusolver.so.12`.

## Những gì đã chỉnh (vì sao và để làm gì)

### 1. Bỏ qua `bpy`, `flash_attn`, `spconv`, `torch_scatter`, `open3d`

- `bpy` (Blender Python API): Blender Foundation không phát hành wheel cho
  ARM64 Linux. Thay thế bằng trimesh để load `.glb`/`.obj`/`.gltf`.
- `flash_attn`: cần compile từ source cho sm_120, hàng giờ. Thay bằng
  PyTorch SDPA (`_attn_implementation: sdpa`).
- `spconv`, `torch_scatter`: cũng cần compile từ source. Chỉ dùng cho
  skin model (PTv3) — không cần cho skeleton prediction nên làm thành
  import lazy (`try/except: None`) trong
  `src/model/{unirig_skin,pointcept/models/PTv3Object,pointcept/models/modules,pointcept/models/utils/structure}.py`.
- `open3d`: không có wheel aarch64. Trong code inference chỉ dùng làm
  fallback cho export — đã có `try/except ImportError` nên tự skip.
- `torch_cluster.fps`: thay bằng FPS pure-PyTorch trong
  `src/model/michelangelo/models/tsal/sal_perceiver.py`.

### 2. Trimesh-based preprocessor

File mới: `preprocess_trimesh.py`. Thay thế bước
`python -m src.data.extract` (cần bpy) bằng:

```bash
python preprocess_trimesh.py --input <file.glb|.obj> \
    --output_dir <npz_dir> --faces_target_count 50000
```

Lưu `raw_data.npz` đúng path mà `run.py` tìm (mirror logic
`get_files`: strip extension, ghép với `output_dir`).

### 3. Lazy `bpy` stub trong `src/data/extract.py`

Để `from src.data.extract import get_files` import được khi không có
`bpy`. Khi code chạm `bpy.*` thì raise lỗi rõ ràng, không crash import.

### 4. Cấu hình

- `configs/model/unirig_ar_350m_1024_81920_float32.yaml`:
  `_attn_implementation: flash_attention_2` → `sdpa`.
- `configs/task/quick_inference_skeleton_articulationxl_ar_256_nofbx.yaml`:
  copy từ `_256.yaml`, tắt `export_fbx` (vẫn giữ `export_obj`,
  `export_npz`).
- `src/system/ar.py`: trong `write_on_batch_end`, export `.obj` skeleton
  cả khi `user_mode=True` (chỉ cần `output_dir`), và dùng `trim=True`
  để path output không bị ghép thêm `npz_dir/`.

## Cách chạy

```bash
# Activate env (Python 3.12 để dùng được cùng Blender nếu sau này cần bpy)
conda activate unirig312
export PYTHONNOUSERSITE=1   # tránh torch cpu-only trong ~/.local che mất env

# Skeleton prediction
cd /home/braitoli/workspace/namnh/code/poc/UniRig
bash run_skeleton.sh examples/giraffe.glb results/giraffe

# Hoặc gọi thẳng 2 bước
python preprocess_trimesh.py --input examples/giraffe.glb \
    --output_dir tmp_skel --faces_target_count 50000
python run.py \
    --task=configs/task/quick_inference_skeleton_articulationxl_ar_256_nofbx.yaml \
    --input=examples/giraffe.glb \
    --output_dir=results/giraffe \
    --npz_dir=tmp_skel \
    --seed=12345
```

## Giới hạn / chưa làm được

- **Skin prediction** (`bash launch/inference/generate_skin.sh`) cần
  `spconv` + `torch_scatter` + `flash_attn` thật. Mỗi cái phải compile
  từ source cho sm_120 — có thể mất hàng giờ mỗi cái. Hiện tại
  import đã được làm lazy (không crash khi thiếu), nhưng để chạy được
  sẽ cần build từ source.
- **FBX export**: vẫn dùng bpy. Khi Blender Foundation phát hành wheel
  ARM64 (chưa có kế hoạch), bật lại `export_fbx: skeleton` trong
  task config.
- **VRM / FBX / DAE input**: bỏ qua — chỉ hỗ trợ `.glb`/`.gltf`/`.obj`
  qua trimesh.
- **Merge step** (`launch/inference/merge.sh`): cần bpy để ráp
  skeleton vào mesh gốc. Có thể thay bằng trimesh nhưng chưa cần
  vì user chỉ cần skeleton.
