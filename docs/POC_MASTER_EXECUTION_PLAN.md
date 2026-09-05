# KẾ HOẠCH TRIỂN KHAI CHI TIẾT (MASTER EXECUTION PLAN)
## POC TỐI ƯU HÓA HẠ TẦNG VAST.AI CHO TRELLIS.2-4B VÀ PIXAL3D TRÊN RTX 4090

> **Trạng thái:** Sẵn sàng thực thi  
> **Ngân sách khả dụng:** $10.00 USD (Dự kiến tiêu hao: ~$2.50 – $2.75 USD)  
> **Phần cứng mục tiêu:** NVIDIA GeForce RTX 4090 (24GB VRAM, Ada Lovelace, SM 8.9)  
> **Văn bản này đóng vai trò:** Checklist thực thi từng bước (Actionable Runbook & Tracker).

---

## MỤC LỤC
1. [Bộ tiêu chí nghiệm thu (Acceptance Criteria)](#1-bộ-tiêu-chí-nghiệm-thu)
2. [Tập dữ liệu chuẩn phục vụ Benchmark](#2-tập-dữ-liệu-chuẩn-phục-vụ-benchmark)
3. [Quy trình thực thi 6 giai đoạn](#3-quy-trình-thực-thi-6-giai-đoạn)
   - [Giai đoạn 0: Chuẩn bị Cục bộ](#giai-đoạn-0-chuẩn-bị-cục-bộ-local-prep)
   - [Giai đoạn 1: Khởi tạo Máy chủ Test RTX 4090](#giai-đoạn-1-khởi-tạo-máy-chủ-test-rtx-4090)
   - [Giai đoạn 2: Cài đặt & Đóng gói Wheels CUDA](#giai-đoạn-2-cài-đặt--đóng-gói-wheels-cuda)
   - [Giai đoạn 3: Thực nghiệm & Đo đạc TRELLIS.2-4B](#giai-đoạn-3-thực-nghiệm--đo-đạc-trellis2-4b)
   - [Giai đoạn 4: Thực nghiệm & Đo đạc Pixal3D](#giai-đoạn-4-thực-nghiệm--đo-đạc-pixal3d)
   - [Giai đoạn 5: Thực nghiệm Kiến trúc Song song (Distributed 2x 1-GPU)](#giai-đoạn-5-thực-nghiệm-kiến-trúc-song-song-distributed-2x-1-gpu)
   - [Giai đoạn 6: Tổng hợp Kết quả, Unit Economics & Dọn dẹp](#giai-đoạn-6-tổng-hợp-kết-quả-unit-economics--dọn-dẹp)
4. [Nhật ký thực thi & Bảng theo dõi tiến độ](#4-nhật-ký-thực-thi--bảng-theo-dõi-tiến-độ)

---

## 1. BỘ TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

Mỗi giai đoạn chỉ được coi là hoàn tất khi thỏa mãn các điều kiện tiên quyết:

- [ ] **AC-1 (Zero-compile Wheels):** Tất cả các thư viện C++/CUDA (`natten`, `flash-attn`, `cumesh`, `o_voxel`, `nvdiffrast`) được biên dịch thành công thành file `.whl` và có thể tái sử dụng ngay trong các máy sau mà không cần compile lại.
- [ ] **AC-2 (Model Benchmark Complete):** Thu thập đầy đủ số liệu thực tế về VRAM đỉnh (Peak VRAM), GPU utilization, và độ trễ (latency) của cả Trellis 2 và Pixal3D trên RTX 4090 ở độ phân giải tiêu chuẩn (1024) và cao (1536).
- [ ] **AC-3 (Concurrency Boundary Clear):** Xác định dứt khoát giới hạn tiến trình trên 1 card 24GB (có chạy an toàn 2 tiến trình hay bắt buộc 1 GPU = 1 tiến trình).
- [ ] **AC-4 (Cost Model Verified):** Có công thức tính chính xác chi phí USD trên 1.000 file 3D hoàn chỉnh.
- [ ] **AC-5 (Budget Protection):** Tổng chi phí thực hiện PoC được kiểm soát chặt chẽ dưới $3.00 USD, máy chủ được hủy (destroy) ngay sau khi hoàn thành đo đạc.

---

## 2. TẬP DỮ LIỆU CHUẨN PHỤC VỤ BENCHMARK

Để kết quả đo đạc khách quan và có tính đại diện, đợt test sử dụng 3 ảnh đầu vào đại diện cho 3 độ phức tạp hình học khác nhau:
1. **Sample A (Vật thể cơ bản - Organic/Simple):** Quả táo hoặc cốc cà phê (lưới ít góc cạnh, texture đơn giản).
2. **Sample B (Vật thể cứng cáp - Hard-surface):** Xe hơi thể thao hoặc máy bay mô hình (nhiều góc nhọn, phản xạ kim loại, cấu trúc PBR phức tạp).
3. **Sample C (Nhân vật/Sinh vật - Full Character/Anatomy):** Tượng nhân vật dạng T-pose/A-pose có nếp gấp quần áo và chi tiết khuôn mặt.

---

## 3. QUY TRÌNH THỰC THI 6 GIAI ĐOẠN

### Giai đoạn 0: Chuẩn bị Cục bộ (Local Prep)
*Thời gian dự kiến: 10 phút | Chi phí: $0.00*

- [ ] **Bước 0.1:** Tạo thư mục lưu trữ mã nguồn và kết quả PoC tại `scripts/poc_3d/`.
- [ ] **Bước 0.2:** Tải/chuẩn bị 3 ảnh test chuẩn (`sample_simple.png`, `sample_hardsurface.png`, `sample_character.png`).
- [ ] **Bước 0.3:** Viết script theo dõi tài nguyên phần cứng liên tục (`gpu_monitor.py` - ghi log VRAM, nhiệt độ, công suất mỗi 0.5 giây).

---

### Giai đoạn 1: Khởi tạo Máy chủ Test RTX 4090
*Thời gian dự kiến: 10 phút | Chi phí: ~$0.05*

- [ ] **Bước 1.1:** Dùng `vastai search offers` quét tìm máy chủ **1x RTX 4090 Verified** có:
  * Đơn giá On-demand $\le \$0.33$/h.
  * Tốc độ mạng Download $\ge 500$ Mbps.
  * Dung lượng ổ đĩa khả dụng $\ge 60$ GB.
  * Vị trí gần (ưu tiên VN, KR, JP, US).
- [ ] **Bước 1.2:** Khởi tạo instance với image chuẩn `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`, mở cổng SSH trực tiếp (`--direct`).
- [ ] **Bước 1.3:** Kết nối SSH, kiểm tra driver NVIDIA, CUDA version (`nvidia-smi`), băng thông internet thực tế.

---

### Giai đoạn 2: Cài đặt & Đóng gói Wheels CUDA
*Thời gian dự kiến: 45 phút | Chi phí: ~$0.25*

- [ ] **Bước 2.1:** Cài đặt các công cụ build cơ bản (`ninja-build`, `git`, `build-essential`).
- [ ] **Bước 2.2:** Build & Install `natten==0.21.0` theo đúng bản PyTorch 2.4 + CUDA 12.4.
- [ ] **Bước 2.3:** Build & Install `flash-attn`.
- [ ] **Bước 2.4:** Build & Install các sub-wheels đặc thù:
  * `cumesh`
  * `o_voxel`
  * `nvdiffrast`
  * `flex_gemm`
- [ ] **Bước 2.5:** Đóng gói toàn bộ các file `.whl` vừa biên dịch vào thư mục `/workspace/prebuilt_wheels/` và lưu bản backup để phục vụ khởi động nhanh cho mọi máy sau này.

---

### Giai đoạn 3: Thực nghiệm & Đo đạc TRELLIS.2-4B
*Thời gian dự kiến: 45 phút | Chi phí: ~$0.25*

- [ ] **Bước 3.1:** Clone repository `microsoft/TRELLIS.2` và tải trọng số mô hình 4B từ Hugging Face.
- [ ] **Bước 3.2:** Khởi chạy `gpu_monitor.py` ở chế độ ngầm.
- [ ] **Bước 3.3:** **Test Case TC-01 (1024 Resolution, 1 Worker):**
  * Chạy lần lượt 3 ảnh mẫu.
  * Ghi lại: Thời gian load weights, thời gian sinh (seconds), Peak VRAM (MB).
- [ ] **Bước 3.4:** **Test Case TC-02 (1536 Resolution, 1 Worker):**
  * Chạy lần lượt 3 ảnh mẫu ở độ phân giải tối đa.
  * Ghi lại: Peak VRAM, thời gian sinh, tỷ lệ hoàn thành mesh.
- [ ] **Bước 3.5:** **Test Case TC-03 (Stress Test Concurrency 2 Workers / 1 GPU):**
  * Spawn đồng thời 2 tiến trình độc lập cùng trỏ vào GPU 0 ở độ phân giải 1024.
  * Ghi lại: Có xảy ra OOM hay không? Nếu thành công, tổng thời gian bị chậm đi bao nhiêu %?
- [ ] **Bước 3.6:** Lưu lại output file 3D (`.glb`) mẫu để đánh giá chất lượng visual.

---

### Giai đoạn 4: Thực nghiệm & Đo đạc Pixal3D
*Thời gian dự kiến: 45 phút | Chi phí: ~$0.25*

- [ ] **Bước 4.1:** Clone repository `TencentARC/Pixal3D` và tải trọng số mô hình từ Hugging Face.
- [ ] **Bước 4.2:** **Test Case TC-04 (Standard Mode, 1 Worker):**
  * Chạy lần lượt 3 ảnh mẫu ở độ phân giải 1024.
  * Ghi lại: Peak VRAM, thời gian sinh.
- [ ] **Bước 4.3:** **Test Case TC-05 (Low-VRAM Mode `--low_vram`, 1 Worker):**
  * Chạy 3 ảnh mẫu với cờ `--low_vram` (CPU offloading theo từng stage).
  * Ghi lại: Mức giảm VRAM thực tế (giảm được bao nhiêu GB?), thời gian sinh bị chậm đi bao nhiêu giây?
- [ ] **Bước 4.4:** **Test Case TC-06 (Stress Test Concurrency 2 Workers / 1 GPU):**
  * Chạy 2 tiến trình Pixal3D ở chế độ `--low_vram` đồng thời trên GPU 0.
  * Ghi lại: Tính khả thi khi chạy song song 2 worker trên card 24GB.
- [ ] **Bước 4.5:** Lưu lại output file 3D (`.glb`) để so sánh đối đầu với Trellis 2.

---

### Giai đoạn 5: Thực nghiệm Kiến trúc Song song (Distributed 2x 1-GPU)
*Thời gian dự kiến: 30 phút | Chi phí: ~$0.32*

- [ ] **Bước 5.1:** Thuê thêm máy **1x RTX 4090 thứ 2**.
- [ ] **Bước 5.2:** Áp dụng gói `prebuilt_wheels` từ Giai đoạn 2 vào máy thứ 2. Đo chính xác thời gian chuẩn bị môi trường (Cold-start time).
- [ ] **Bước 5.3:** Thiết lập kịch bản phân tải đơn giản (Queue dispatch 10 ảnh ngẫu nhiên chia đều cho 2 máy).
- [ ] **Bước 5.4:** Ghi nhận: Throughput tổng (số model/phút), tính ổn định và chi phí phát sinh thực tế.

---

### Giai đoạn 6: Tổng hợp Kết quả, Unit Economics & Dọn dẹp
*Thời gian dự kiến: 15 phút | Chi phí: $0.00*

- [ ] **Bước 6.1:** Hủy (destroy) toàn bộ các instance trên Vast.ai qua lệnh `vastai destroy instance <id> -y` để chấm dứt tính cước.
- [ ] **Bước 6.2:** Tổng hợp số liệu vào bảng so sánh đối đầu (VRAM, Latency, Throughput, Cost/Asset).
- [ ] **Bước 6.3:** Xuất bản tài liệu Playbook hướng dẫn triển khai Production và bộ script tự động hóa hạ tầng.

---

## 4. NHẬT KÝ THỰC THI & BẢNG THEO DÕI TIẾN ĐỘ

| Giai đoạn | Nội dung chính | Trạng thái | Thời gian thực tế | Chi phí thực tế | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Giai đoạn 0** | Chuẩn bị script & ảnh test | `PENDING` | -- | $0.00 | Chuẩn bị trên máy local |
| **Giai đoạn 1** | Spawn máy 1x 4090 đầu tiên | `PENDING` | -- | -- | Chọn máy VN/KR giá rẻ |
| **Giai đoạn 2** | Build & cache CUDA wheels | `PENDING` | -- | -- | Tạo artifact `.whl` |
| **Giai đoạn 3** | Benchmark Trellis 2 4B | `PENDING` | -- | -- | Test 1024, 1536 & 2 workers |
| **Giai đoạn 4** | Benchmark Pixal3D | `PENDING` | -- | -- | Test standard vs low_vram |
| **Giai đoạn 5** | Test phân tán 2 máy 4090 | `PENDING` | -- | -- | Đo cold-start & throughput |
| **Giai đoạn 6** | Báo cáo, Unit Economics & Hủy | `PENDING` | -- | -- | Chốt kết quả & đóng máy |
