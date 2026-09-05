# 📘 RUNBOOK HƯỚNG DẪN CÀI ĐẶT & KINH NGHIỆM TRIỂN KHAI CLOUD GPU (TRELLIS 2 & PIXAL3D)

Tài liệu này tổng hợp toàn bộ quy trình, script tự động, kinh nghiệm xử lý lỗi (troubleshooting) và nguyên tắc chọn máy chủ chi phí thấp nhất khi triển khai hệ thống tạo mô hình 3D (Tô tượng 3D / 3D Statue Studio).

---

## 1. NGUYÊN TẮC CHỌN MÁY CHỦ CLOUD (VAST.AI / RUNPOD) TỐI ƯU CHI PHÍ

1. **Bộ lọc tìm kiếm giá rẻ nhất:**
   - Luôn sắp xếp theo đơn giá tăng dần (`dph_total` hoặc `dph`).
   - Kiểm tra cờ `"rentable": true` (loại bỏ các máy hiển thị giá rẻ ảo nhưng thực chất đã bị thuê kín hoặc offline).
   - Kiểm tra băng thông PCIe (`pcie_bw >= 8.0 GB/s`). Tránh các máy PCIe 1.5 GB/s (1x) vì sẽ làm nghẽn cổ chai thời gian nạp trọng số 20-30GB vào VRAM.
   - **Dung lượng ổ cứng (Disk Size):** Luôn cấu hình tối thiểu **150GB – 200GB** (thay vì 50-60GB). Các model 3D SOTA (Pixal3D 26GB, Trellis 2 17GB, cache, CUDA build temp) chiếm hơn 45GB. Giá thuê thêm ổ cứng trên Vast.ai rất rẻ (chỉ ~$0.001/GB/tháng, 200GB tốn chưa đến $0.01/ngày) nhưng tránh được hoàn toàn rủi ro Disk Full (OOD).
   - Lệnh tìm kiếm tối ưu trên Vast.ai:
     ```bash
     vastai search offers 'reliability > 0.90 num_gpus = 1 gpu_name = RTX_4090 rentable = true' -o 'dph_total'
     ```

2. **Lựa chọn Base Docker Image:**
   - ❌ **Tránh dùng:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` vì mặc định **không có SSH daemon** (`sshd`), dẫn đến instance không thể truy cập qua SSH.
   - ✅ **Nên dùng:** `vastai/pytorch:cuda-12.4.1-auto` (đã tích hợp SSH daemon, PyTorch 2.4+, CUDA 12.4/12.6, môi trường ảo tại `/venv/main`).

---

## 2. QUY TRÌNH BIÊN DỊCH VÀ CÀI ĐẶT EXTENSIONS CUDA

### Danh mục thư viện và cờ biên dịch chuẩn:

| Thư viện | Mục đích | Lưu ý kỹ thuật & Lỗi thường gặp |
| :--- | :--- | :--- |
| **`nvdiffrast`** | Differentiable rasterization cho render mesh & bake texture | **BẮT BUỘC** thêm cờ `--no-build-isolation` để compiler nhìn thấy `torch.utils.cpp_extension` và CUDA toolkit. |
| **`o_voxel`** | Thư viện biểu diễn Sparse Voxel & Octree | Chứa 2 sub-package quan trọng: `cumesh` và `flex_gemm`. |
| **`cumesh`** | Xử lý hình học mesh cực nhanh trên CUDA | Cần clone submodule third-party (`nanobind`): `git submodule update --init --recursive`. |
| **`flex_gemm`** | Sparse Matrix Multiplication | Thay thế hoàn toàn cho `spconv` (vốn không tương thích Python 3.12). |
| **`flash-attn`** | Tăng tốc Transformer Attention | Biên dịch bằng nvcc với `--no-build-isolation`. |
| **`natten`** | Neighborhood Attention | **KHÔNG CẦN THIẾT** cho TRELLIS 2 / Pixal3D (chỉ là tàn dư file requirements demo). Bỏ qua để tiết kiệm 10 phút compile. |

---

## 3. SCRIPT TỰ ĐỘNG CÀI ĐẶT 1-CLICK

Tất cả đã được đóng gói trong file [`scripts/poc_3d/setup_all_dependencies.sh`](file:///home/braitoli/workspace/namnh/code/poc/UniRig/scripts/poc_3d/setup_all_dependencies.sh):
```bash
# Thực thi trên máy chủ mới:
bash scripts/poc_3d/setup_all_dependencies.sh
```

### Kinh nghiệm tải trọng số Model lớn (17GB - 26GB):
- Tuyệt đối không chạy lệnh tải model trực tiếp qua SSH session thông thường vì rớt mạng SSH (timeout 255) sẽ ngắt tiến trình.
- Luôn chạy trong `tmux` session nền:
  ```bash
  tmux new-session -d -s model_download "huggingface-cli download microsoft/TRELLIS.2-4B && huggingface-cli download TencentARC/Pixal3D"
  ```

---

## 4. BẢNG ĐỐI SOÁNH PHẦN CỨNG: MÁY GX10 (NVIDIA GB10) VS RTX 4090

| Tiêu chí | Máy Trạm Cục Bộ GX10 (NVIDIA GB10) | Cloud RTX 4090 (Vast.ai) |
| :--- | :--- | :--- |
| **Kiến trúc GPU** | **NVIDIA Blackwell GB10** | **NVIDIA Ada Lovelace (AD102)** |
| **Kiến trúc CPU** | ARM64 (Cortex-X925 / Cortex-A725, 20 vCPU) | x86_64 (AMD EPYC 7742, 25.6 vCPU) |
| **Bộ nhớ VRAM** | **Unified Memory (128GB+)** chia sẻ CPU-GPU | **24 GB GDDR6X** riêng biệt |
| **Băng thông bộ nhớ** | Rất lớn, nạp model không sợ OOM | 1,008 GB/s (tốc độ xử lý tensor cực nhanh) |
| **Tốc độ Trellis Raw** | **~210.65s (3.5 phút)** *(đo thực tế trên container inference-3d)* | Dự kiến **~18s - 35s** (nhanh gấp **6 - 10 lần**) |
| **Pipeline Tô tượng Full** | ~240s - 260s | Dự kiến **~45s - 65s** |
| **Điểm nghẽn chính** | Tốc độ tính toán thuần FP16/BF16 của chip GB10 thấp hơn AD102; môi trường ARM64 kén wheel pre-built | Giới hạn VRAM 24GB (cần kiểm soát bake texture và resolution) |
| **Chi phí vận hành** | Chi phí máy vật lý cục bộ cố định | **~$0.319 / giờ** (~8.000 VNĐ/giờ) |

---

## 5. CƠ CHẾ ĐÓNG GÓI PRE-BUILT DOCKER IMAGE CHO LẦN SAU

Để lần sau **không phải chờ 10 phút biên dịch** CUDA extensions:
1. Sau khi build xong instance hiện tại, commit container thành Docker Image trên Docker Hub:
   ```bash
   docker commit <container_id> braitoli/statue-3d-runner:cu124-torch24
   docker push braitoli/statue-3d-runner:cu124-torch24
   ```
2. Các lần thuê Vast.ai sau này chỉ cần truyền image `braitoli/statue-3d-runner:cu124-torch24` -> Máy sẵn sàng chạy chỉ trong **60 giây**!
