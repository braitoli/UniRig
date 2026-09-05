# Handoff: UniRig Statue Studio — sửa GLB export, viewer, OOM Pixal3D, phân vùng tô màu (2026-09-04)

## 0. Nguyên tắc hay quên — ĐỌC TRƯỚC

**Mục tiêu tối thượng của dự án**: UniRig Statue Studio (`http://localhost:7860/statue`) là xưởng
tạo tượng 3D từ ảnh 2D (TRELLIS / Hunyuan3D / Pixal3D) rồi xuất nhiều biến thể GLB
(plaster/segmented/id_colored/textured/shell/shell_optimized/package_zip) sẵn sàng cho một app
tô tượng online bên ngoài — file phải mở đúng màu ở mọi viewer (three.js lẫn macOS
Preview/Blender), mesh không được rách, và các vùng đổ màu phải khớp hợp lý với đối tượng.

**3 nguyên tắc hay bị quên nhất** (chép nguyên văn từ CLAUDE.md/memory lúc viết handoff này):

1. **Điều phối/delegate**: `~/.claude/CLAUDE.md` Step 3a — *"Before every delegation ... the main
   agent MUST read and follow `~/.claude/skills/cli-agent-routing/SKILL.md`"*, và luôn truyền
   `model` khi spawn subagent (không truyền thì thừa kế Opus của main agent, kể cả việc thuần cơ
   học). **Trong phiên này**, người dùng ra lệnh trực tiếp *"cho các subagent làm đi, mainagent
   chỉ điều phối"* — mọi việc đã đi qua Agent tool (Claude subagent, `general-purpose`, model
   `sonnet` cho việc bounded / `opus` cho việc cần suy luận đúng-sai không tự lộ ra), **không**
   qua CLI agent ngoài (agy/codex/omp). Phiên sau nên tự đối chiếu lại với
   `cli-agent-routing/SKILL.md` xem cách làm này có đúng bảng quyết định chuẩn không — chưa kiểm
   chứng điều đó trong phiên.

2. **Tài nguyên vật lý rảnh**: máy chạy dự án là **NVIDIA GB10 (DGX Spark), bộ nhớ hợp nhất** —
   GPU cấp phát từ chính RAM hệ thống, pool chung 121,63 GB
   (xem memory `gb10-unified-memory-memfree.md`). Có một tiến trình **TRELLIS ở cổng 7870 dùng
   CHUNG cho cả dự án này lẫn một dự án khác** (memory `trellis-worker-shared-across-projects.md`)
   — **tuyệt đối không `kill` nó, không tự dựng bản TRELLIS mới**. Swap 16 GB thường xuyên gần
   cạn (từng chỉ còn 24-45 MB trống) — tránh chạy nhiều việc nặng (chromium headless, backfill,
   sinh 3D) song song.

3. **Song song tối đa**: phiên này chạy tới 6 subagent song song cùng lúc
   (`pedestal-ui`, `rebuild-assets`, `mesh-holes`, `coplanar-merge`, `history-gone`, `pixal-oom`).
   Nguyên tắc áp dụng: luôn hỏi "việc này có thể chạy đồng thời không" trước khi làm tuần tự,
   nhưng **hai agent không được sửa cùng một file cùng lúc** (đã có sự cố: một agent sửa
   `statue.html`+`statue.js` bằng 2 lần ghi tách rời làm lộ khoảng trống gây lỗi cho người dùng
   thật đang mở trang) — điều phối bằng cách xếp hàng việc trên cùng file, không giao song song.

## 1. Mục tiêu phiên này

Bắt đầu từ câu hỏi "sao tượng tải về không có màu" ở `/statue`, mở rộng dần thành: sửa toàn bộ
pipeline xuất GLB (`pipeline/statue_optimizer.py`), sửa UI viewer 3D (`playground/static/statue.*`),
thêm Pixal3D làm generator thứ ba, sửa lỗi OOM khi chọn Pixal3D, và cải thiện thuật toán phân vùng
tô màu (từ lát cắt ngang cứng sang phân cụm theo màu texture).

## 2. Đã xong, ĐÃ VERIFY (kết quả chuẩn)

**Pipeline xuất GLB** (`pipeline/statue_optimizer.py`, đã commit qua `d7eb736`→`4bc055c`):
- Hàn đỉnh theo vị trí trước khi decimate `shell_optimized` — diện tích giữ được 61%→99,2%.
- Chiếu ngược UV từ mesh nguồn bằng barycentric không kẹp (bug của `trimesh.points_to_barycentric`
  kẹp về [0,1] làm texture bết) — mặt lỗi UV 13,98%→0,01%.
- Hàn góc UV theo dung sai pixel (VTOL 0,5px) thay vì làm tròn lưới — 61.214→36.511 đỉnh (−40%).
- Texture xuất JPEG thay WebP (WebP nằm trong `extensionsRequired`, macOS Preview bỏ texture →
  tượng ra kim loại đen); `metallicFactor=0`, `roughnessFactor` đo từ map thật (không ép fallback
  cứng — bug tự phát hiện: 3 biến thể cùng job phải cùng giá trị).
- `add_statue_pedestal`: xoay cylinder 90° quanh X (trimesh dựng theo Z, tượng Y-up) — 4 kiểu đế
  đều nằm ngang, đáy đúng Y=0.
- `auto_ground_and_orient` `flatten_bottom`: chỉ ép lớp đáy NGOÀI (bỏ đỉnh có mặt kề hướng lên),
  không ép cả 2 lớp vỏ rỗng về cùng y=0 nữa — mặt bẹp hướng lên 15.198→170 trên job đã chạy lại
  full pipeline (chỉ 1/6 job đủ mẫu vì 5 job kia chỉ backfill nên `plaster.glb` vẫn bản cũ).
- `_strip_dead_uv`: bỏ UV khi material không dùng texture map nào (plaster/segmented) — file
  giảm 15,5-15,6%, `baseColorFactor` không đổi.
- **Nghiệm thu độc lập** (agent `mesh-holes`, không biết agent thực thi làm gì): 6 job, 0/6
  boundary edge, render FrontSide diff 0,07-0,10% so với đối chứng dương 1,68% (mesh rách code
  cũ) — thước đo tự chứng minh phân biệt được trước khi dùng để chấm.
- Root cause gốc của "mesh xé rách" là do **server chạy `uvicorn` không có `--reload`**, sửa code
  buổi sáng không có tác dụng cho tới lần restart đầu tiên lúc 12:11 — đã chứng minh bằng cách
  build lại code CŨ và ra số liệu trùng khớp file lỗi từng byte.

**Giao diện `/statue`** (`playground/static/statue.{html,css,js}`):
- Sửa kẹt trắng khi quay lại chế độ "Texture AI Gốc" (nhánh `textured` chỉ tắt wireframe thay vì
  khôi phục vật liệu gốc từ `originalMaterialsMap`).
- Sửa bug **xoay model tự nhảy về "Đang Tô Màu"**: `onCanvasPointerDown` ép đổi chế độ vô điều
  kiện trên MỌI `pointerdown`, kể cả bước đầu kéo xoay camera. Đã bỏ hẳn click-to-paint trên
  canvas, bỏ nút chế độ "Đang Tô Màu" và nút tải "Tượng Đã Tô Màu Trực Tiếp" theo quyết định của
  người dùng — tô màu thật vẫn còn ở tab "Studio 3DPainting" riêng.
- Sửa chế độ "Phân Vùng Đổ Màu" hiện sai file (`currentGlbUrl.includes('_textured.glb')` không
  bao giờ khớp URL job vì job không có đuôi `.glb`) — giờ nạp đúng `segmented_glb`, vẽ đường viền
  ranh giới 7 vùng (`LineSegments` per submesh, cache theo model). Commit `4695c7c`.
- Preview 3D khi hover nút tải (render bằng three.js dùng chung, cache theo URL).
- Thêm Pixal3D vào danh sách generator; mặc định Decimation đổi sang "Nguyên bản"; "Không Có Đế"
  chuyển lên đầu; bỏ ô "Tích hợp khung xương & Hoạt ảnh".
- Mở job từ lịch sử giờ mặc định nạp `textured_glb` (fallback về `segmented_glb` nếu HEAD 404
  hoặc GLTFLoader lỗi parse — verify bằng 2 test cô lập).
- Sửa lỗi lịch sử tạo tượng "biến mất": `DOMContentLoaded` chạy tuần tự không try/catch, một lỗi
  null ở `initPalette()` (do 2 file sửa lệch thời điểm) chặn đứng `loadStatueHistory()` phía sau.
  Đã bọc try/catch từng bước độc lập.

**Phân vùng tô màu** (2 commit riêng, đã duyệt qua `AskUserQuestion`):
- `d1bbca4`: đổi nhãn từ giải phẫu người (Đầu/Tóc/Tay...) sang trung tính "Vùng 1".."Vùng 7" —
  vì nhãn cũ áp cho xe hơi/nhà nấm vô nghĩa. **Không xoá** `STATUE_PALETTE[6]` (mục chết) vì tra
  cứu theo `pid % len`, xoá sẽ âm thầm đổi màu đế từ xám sang đỏ. Sửa luôn bug có sẵn: tên node
  cũ `Part_00_Đầu` không khớp text sidebar `Đầu / Khuôn mặt (Head)` nên hover-highlight chưa từng
  hoạt động — giờ tên node = tên nhãn nên khớp.
- `4bc055c`: phân vùng theo **màu texture** (k=7 cố định trong không gian Lab qua UV) thay lát
  cắt Y khi vật có màu đủ tách biệt; cổng chặn bằng độ trải chroma (ngưỡng 10,0 — xe 1,63 vs
  robot 18,28, cách 11 lần). Vật đơn sắc hoặc không texture rơi về nhánh hình học (Y-slice) cũ.
  Verify đầu-cuối qua `Statue3DPipeline.process_statue` thật (63,4s, không tốn GPU): log in đúng
  `MÀU TEXTURE: 7 cụm`, 7 node tên `Vùng 1..7`, 14/14 test xanh. Lý do quyết định: đo trên 5
  tượng, độ trải Y ranh giới tăng từ 0,002 (mặt phẳng chết cứng) lên ~0,25 (bám vật thật, gấp
  125-140 lần), ổn định 4/5 vật (ARI 0,867-0,986).

**Sự cố đã xử lý**: `git checkout main` + `git merge --ff-only` sau khi `git rm --cached` 30 file
preset GLB trên nhánh riêng đã **xoá sạch chúng khỏi đĩa** (không chỉ khỏi git index). Đã khôi
phục từ `git ls-tree 549092b` rồi `rebuild-assets` dựng lại bằng code đã vá (trừ `fox_girl` vốn
không có nguồn texture từ trước sự cố).

## 3. Đã làm nhưng CHƯA verify / còn nghi ngờ

- **`pedestal-ui`** đang có diff WIP **chưa commit** trong `playground/static/statue.js` (+28
  dòng): rà soát ma trận 4×4 chế độ xem, sửa `originalMaterialsMap` không bị lỗi thời sau khi
  nạp model mới, thêm `loadRequestSeq` chặn race điều kiện ping-pong khi bấm nhanh giữa 2 chế độ
  tự nạp file, `disposeBoundaryLines()` khi đổi model. **Chưa báo cáo hoàn tất, chưa có bằng
  chứng render/headless-chromium cho phần này.**
- **Thanh trượt độ sáng** (đã lỡ commit trong `d7eb736`, code chạy đúng — đo trực tiếp
  `light.intensity` khớp `baseIntensity×factor`) nhưng **phép đo 3 mức 0.4/1.0/2.5x tăng đơn điệu
  trên chế độ Phân Vùng chưa có kết quả cuối** — lần đo trên "Thạch Cao Trắng" bị cháy sáng ACES
  nên không phân biệt được, đang đo lại trên chế độ có màu phẳng thì bị việc khác chen ngang.
- **`pixal-oom`**: đang thực thi thay đổi chính sách vừa chốt lại (bỏ hẳn auto-start worker 7865,
  thêm cấu hình bật lại thường trú) — **chưa xong, chưa verify**. Có 1 job Pixal3D thật đang chạy
  ở 88% lúc viết handoff này (xem mục 4), chưa xác nhận `completed`.
- Giả định "job cũ KHÔNG dựng lại theo phân vùng màu mới, chỉ job mới có" — tôi tự đặt và đã nói
  với `mesh-holes`, **người dùng chưa xác nhận rõ ràng**.

**ĐÃ XONG sau khi viết bản đầu của handoff này** — `mesh-holes` hoàn tất việc so sánh nhánh dự
phòng cho vật đơn sắc (lát cắt Y hiện tại vs k-means toạ độ 3D vs k-means toạ độ+pháp tuyến), đo
trên xe + thỏ + job không texture. **Kết luận: GIỮ NGUYÊN lát cắt Y, không sửa code.** Phương án
pháp tuyến có chỉ số "độ trải Y" cao nhất nhưng ảnh render lại tệ nhất (vùng thành dải mỏng ven
bóng, không đổ màu được — agent tự cảnh báo đây là lần thứ 2 trong phiên một chỉ số trông hợp lý
mà không đo đúng thứ cần đo). Phương án k-means toạ độ không hơn A trên xe và tệ hơn hẳn A trên
thỏ (A: 4 vùng mạch lạc; B: 7 vùng vụn, tai bị xé 2 màu). Ảnh:
`/home/braitoli/cmp_fallback_xe.png`, `/home/braitoli/cmp_fallback_tho.png`. **Không có diff nào
để commit từ việc này** — coi như đã đóng, mục 6 bước 3 (bên dưới) đã xong, không cần chờ nữa.

- **`pixal-oom`**: job Pixal3D thật đã chạy tới `completed` — `statue_1788512135_20260904-150247`,
  686,77s, 977.979 mặt, 7 phân vùng, không lỗi, không hạ tham số chất lượng nào (verify bbox lệch
  <2% giữa chế độ tiêu chuẩn và low_vram). Log server xác nhận cơ chế preflight + tự chuyển
  low_vram + chặn tự dẫm chân đều kích hoạt thật. **NHƯNG** đây là bản build TRƯỚC khi nhận quyết
  định mới nhất của tôi (bỏ hẳn auto-start worker 7865) — log job này vẫn còn dòng
  `⚡ Auto-starting persistent TRELLIS GPU worker on port 7865...` sau khi job xong, tức đường
  "dừng rồi tự dựng lại worker" (gần phương án C) vẫn đang chạy, CHƯA đổi sang phương án A/lai đã
  chốt. Agent đang hỏi lại đúng câu tôi đã trả lời ở lượt trước ("chọn (ii), bỏ hẳn auto-start
  7865") — quyết định đã gửi, có thể agent chưa kịp xử lý vì mải theo dõi job 686s. **Việc còn
  lại**: xác nhận agent đã nhận quyết định, áp nó, rồi chạy lại phép thử
  TRELLIS→Pixal3D→TRELLIS để xác nhận đường mới hoạt động.

## 4. Đang chạy / chưa xong

- **`pixal-oom`** (agent, trạng thái `running`): sửa chính sách worker TRELLIS theo quyết định
  mới nhất — bỏ hẳn auto-start ở cổng 7865, thêm cấu hình kiểu `STATUE_RESIDENT_GENERATOR` để bật
  lại thường trú cho 1 model khi cần sinh liên tục, và kiểm xem cổng 7870 (TRELLIS dùng chung) có
  tái sử dụng được thay vì dựng worker riêng không. Diff uncommitted: `pipeline/pixal3d_generator.py`
  (+450 dòng), `playground/server.py` (+50/−39 dòng).
  - Kiểm tiến độ job Pixal3D đang chạy:
    ```bash
    cd /home/braitoli/workspace/namnh/code/poc/UniRig
    PYTHONNOUSERSITE=1 /home/braitoli/miniconda/envs/unirig312/bin/python -c "
    import sys; sys.path.insert(0,'playground'); import database
    for j in database.list_statue_jobs(limit=6): print(j['id'], j['status'], j.get('metadata',{}).get('progress',{}))"
    ```
    Job cần theo dõi: `statue_1788512135_20260904-150247` (đang 88% lúc viết handoff).
  - Resume: `SendMessage({to: "pixal-oom", message: "..."})` hỏi trạng thái.
- **`mesh-holes`** (agent, trạng thái `running`): so sánh nhánh dự phòng cho vật đơn sắc (xe).
  Resume: `SendMessage({to: "mesh-holes", message: "..."})`.
- **`pedestal-ui`** (agent, `idle` nhưng còn WIP chưa report) — xem mục 3. Resume tương tự.
- **Server**: `playground/server.py` chạy PID xác nhận gần nhất **1048627**, khởi động
  `2026-09-04 15:48:34`. Uvicorn **không có `--reload`** — mọi sửa `pipeline/*.py` hay
  `playground/server.py` cần restart thủ công mới có hiệu lực. Cách restart an toàn đã dùng suốt
  phiên:
  ```bash
  cd /home/braitoli/workspace/namnh/code/poc/UniRig
  pgrep -f "playground/server.py"           # xác nhận đúng PID trước khi kill
  kill <PID>                                 # kill thường, KHÔNG -9
  sleep 4
  PYTHONNOUSERSITE=1 nohup /home/braitoli/miniconda/envs/unirig312/bin/python playground/server.py \
    > playground/server.log 2>&1 &
  # poll: curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/statue  (đợi 200)
  ```
  **CẢNH BÁO**: không dùng `pkill -f "<pattern chuỗi thô>"` — đã lỡ dùng
  `pkill -f "trellis.*7865"` và nó khớp trúng chính tiến trình shell đang gõ lệnh, làm sập cả
  server. Luôn kill theo PID cụ thể đã xác nhận qua `ps`.

## 5. Quyết định đã chốt

- Preset GLB (`playground/static/sample_presets/models/`) **không commit vào git** (đã thêm
  `.gitignore`), dựng lại bằng `backfill_statue_shells.py` — người dùng xác nhận qua
  `AskUserQuestion`.
- Merge thẳng vào `main`, không giữ nhánh riêng — người dùng: *"merge luôn đi, không cần tách
  nhánh đâu"*. Mọi commit từ giờ đi thẳng `main`.
- Bỏ hẳn công cụ tô màu trên canvas ở tab "Xem 3D" (thừa, gây bug) — tô màu thật chỉ còn ở tab
  "Studio 3DPainting".
- Pixal3D OOM: chọn **load-on-demand** (phương án A), sau đó chốt thêm **bỏ hẳn auto-start worker
  7865**, giữ cấu hình bật lại thường trú cho 1 model khi cần sinh liên tục — đây là quyết định
  MỚI NHẤT, có thể khác với bản `pixal-oom` đã lỡ implement trước đó (phương án gần C).
- Phân vùng: chuyển sang phân cụm theo màu texture (có cổng chặn), nhãn trung tính "Vùng N" —
  cả hai xác nhận qua `AskUserQuestion`.
- Chiếc xe / vật đơn sắc không tách được bánh-kính-đèn (marching-cubes ra 1 vỏ liền 96,2% diện
  tích) — người dùng chấp nhận, nói *"tự tính xem làm sao tốt nhất có thể thôi"* → đang thử thêm
  1 vòng bounded (mục 4).
- Worker TRELLIS cổng 7870 dùng chung dự án khác — **không được kill/dựng bản mới**, đã ghi vào
  memory persistent.

## 6. Việc tiếp theo, đúng thứ tự ưu tiên

1. Chờ `pixal-oom` báo cáo: job Pixal3D 88% có `completed` không; chính sách bỏ auto-start 7865 +
   config thường trú đã áp xong chưa; cổng 7870 có tái dùng được không; chạy đủ phép thử
   TRELLIS→Pixal3D→TRELLIS trong 1 phiên server. Restart server (theo lệnh ở mục 4) sau khi xong,
   rồi commit `pipeline/pixal3d_generator.py` + `playground/server.py`.
2. Chờ `pedestal-ui` báo cáo hoàn tất ma trận 4×4 chế độ xem + fix race condition, verify bằng
   headless chromium, rồi commit `playground/static/statue.js`.
3. ~~Chờ `mesh-holes`~~ ĐÃ XONG: giữ nguyên lát cắt Y, không có gì để commit (xem mục 3).
4. Hỏi lại người dùng: job cũ có cần dựng lại `segmented_glb`/`id_colored_glb` theo phân vùng màu
   mới không (hiện đang giả định KHÔNG).
5. Xoá 4 file tạm còn sót trong `playground/static/`: `tmp_brightness_check.html`,
   `tmp_brightness_debug.html`, `tmp_matrix_check.html`, `tmp_race_check.html` (untracked, vô hại
   nhưng nên dọn trước khi coi là xong).
6. Đối chiếu lại phiên này với `cli-agent-routing/SKILL.md` xem việc dùng toàn Claude subagent có
   đúng bảng quyết định chuẩn không (nêu ở mục 0, chưa kiểm chứng).
7. Sau khi mọi patch xong: restart server lần cuối, chạy nghiệm thu tổng — tối thiểu unittest
   `tests.test_statue_pipeline` xanh, thử tạo 1 tượng thật bằng mỗi generator (TRELLIS/Hunyuan3D/
   Pixal3D) qua UI thật.

## 7. Ranh giới / điều cần hỏi lại người dùng

- Không kill hay dựng thêm tiến trình ở cổng 7870 mà không hỏi trước.
- Không tự ý dựng lại (backfill/regenerate) hàng loạt job cũ — tốn GPU/thời gian, phải hỏi trước.
- Việc restart server: trong phiên có nhiều subagent, main agent giữ quyền restart để tránh cắt
  ngang việc đang chạy — nếu resume ở phiên mới không còn multi-agent song song thì restart bình
  thường theo lệnh ở mục 4, không cần hỏi.

## 8. Trạng thái git/repo + lượt trao đổi gần nhất

```bash
cd /home/braitoli/workspace/namnh/code/poc/UniRig
```
- Branch: `main`. HEAD: `4bc055c` "feat(statue): derive paintable regions from texture colour
  when the model has any".
- `git status`: 3 file **đã sửa nhưng chưa commit** (WIP của agent đang chạy — đừng commit vội):
  `pipeline/pixal3d_generator.py`, `playground/server.py`, `playground/static/statue.js`. 4 file
  untracked vô hại: `playground/static/tmp_brightness_check.html`,
  `tmp_brightness_debug.html`, `tmp_matrix_check.html`, `tmp_race_check.html`.
- 5 commit gần nhất (mới nhất trước):
  ```
  4bc055c feat(statue): derive paintable regions from texture colour when the model has any
  d1bbca4 fix(statue): use neutral region labels instead of anatomical ones
  4695c7c fix(statue): make segmented view load the segmented GLB and outline its regions
  d7eb736 fix(statue): repair GLB export pipeline and 3D viewer in Statue Studio
  4291b31 feat(pixal3d): add Pixal3D generator, multi-view inference and detail presets
  ```
- Thư mục `playground/static/sample_presets/models/` (30 file GLB, ~94-138 MB) đã gitignore, KHÔNG
  còn theo dõi bởi git. Nếu mất lại (ví dụ do thao tác checkout+merge tương tự), khôi phục bằng
  `git checkout 549092b -- playground/static/sample_presets/models/` (bản CŨ, phải chạy lại
  `FORCE_REBUILD=1 backfill_statue_shells.py` để có texture/JPEG đúng) hoặc chạy backfill trực
  tiếp nếu mesh nguồn `stage0_generated/` còn.

**5 lượt trao đổi gần nhất** (tóm sát mạch đang dở):
1. Người dùng hỏi vì sao GLB tải về "tối" hơn preview trong studio → đo và trả lời: chủ yếu do
   ánh sáng renderer (5 đèn + tone-mapping ACES trong studio, chênh 75 lần so với đổi vật liệu) —
   người dùng chấp nhận nhám, chỉ muốn nút chỉnh sáng để tự thử → giao `pedestal-ui` làm thanh
   trượt (đã commit lỡ trong `d7eb736`, đo 3 mức chưa xong hẳn).
2. Người dùng báo bug "xoay model tự nhảy về Đang Tô Màu" → truy đúng nguyên nhân
   (`onCanvasPointerDown` ép mode vô điều kiện) → bỏ hẳn tô màu trên canvas, đã commit.
3. Người dùng hỏi "phân vùng đã chuẩn chưa, đang auto à?" → điều tra sâu, phát hiện lát cắt Y
   cứng không nhìn hình dạng gì cả (bbox giống hệt giữa robot và xe hơi) → đổi nhãn trung tính
   (đã commit) → đổi hẳn sang phân vùng theo màu texture có cổng chặn (đã commit, verify
   đầu-cuối qua pipeline thật).
4. Người dùng hỏi tiếp về xe: "phải chia thành bánh xe, kính xe, thân xe, đèn xe..." → đo bằng
   liên thông, phát hiện marching-cubes ra 1 vỏ liền không tách được → người dùng chấp nhận, nói
   "tự tính xem làm sao tốt nhất có thể" → giao vòng thử bounded cuối (đang chạy).
5. Song song, người dùng báo lỗi CUDA OOM khi chọn Pixal3D → giao `pixal-oom`, nhiều vòng đo/sửa
   thật (2 lần job thất bại → 2 lỗ hổng được vá), người dùng chốt chính sách "khi nào dùng thì
   mới load" rồi làm rõ thêm "để sẵn config bật thường trú khi cần" → vừa nhắc thêm cổng 7870
   dùng chung dự án khác, không được đụng → agent đang áp chính sách cuối, chưa xong. Cuối cùng
   người dùng chạy `/model` (đổi mặc định sang Sonnet 5) rồi gọi `/brai-handoff`.

## 9. Bài học — chỗ đã bị chủ dự án sửa/nhắc trong phiên này

- **`git checkout main` + `git merge --ff-only` sau `git rm --cached` đã xoá 30 file GLB khỏi
  đĩa**, không chỉ khỏi git index → tôi tự phát hiện và khôi phục, không phải do người dùng chỉ
  ra, nhưng ghi lại làm bài học: kiểm working tree thật sau merge, không chỉ nhìn `git log`.
- **`pkill -f "trellis.*7865"` giết nhầm chính tiến trình shell đang chạy nó**, làm sập server
  ngoài ý muốn → tự phát hiện ngay sau đó. Bài học: không dùng `pkill` theo pattern chuỗi thô,
  luôn kill theo PID cụ thể.
- Người dùng phải nhắc **"cho các subagent làm đi, mainagent chỉ điều phối"** và sau đó
  **"chạy lâu quá"** khi thấy main agent tự chạy `grep`/đọc code thay vì giao ngay cho subagent
  → Bài học: khi có nhiều subagent hoạt động, main agent hạn chế tối đa tự thực thi, ưu tiên điều
  phối và tổng hợp.
- Tôi từng đưa ra hàng loạt giả thuyết sai và bị agent `mesh-holes` đo rồi bác bỏ: 4 giả thuyết về
  nguyên nhân mesh rách (H1-H4, thật ra chỉ do server chưa restart), phương án gộp mặt đồng phẳng
  "lợi 11%" (thật ra do bịt lỗ hổng, lợi thật 0%), heuristic "khuôn mặt bị đảo ngược" (tự đính
  chính sai), và giả thuyết model tải về "tối do vật liệu" (sai 75 lần so với do ánh sáng) → đã
  ghi vào memory `verification-metrics-need-controls.md`: luôn đòi bằng chứng đo bằng số kèm đối
  chứng dương trước khi tin một giả thuyết, kể cả giả thuyết của chính mình.
- Tôi quên restart server sau khi sửa `pipeline/statue_optimizer.py` buổi sáng, khiến bản vá
  không có tác dụng suốt nhiều giờ cho tới lần restart đầu tiên → ghi vào memory
  `restart-playground-server-freely.md`: luôn đối chiếu giờ khởi động tiến trình với giờ sửa file
  cuối cùng trước khi tin kết quả.
- Tôi gợi ý sai cho brief của `pixal-oom` rằng hạ "Chất lượng Mesh" có thể cứu OOM — agent đo và
  bác bỏ (OOM xảy ra lúc nạp model, trước khi resolution có ý nghĩa) → Bài học: không đưa gợi ý
  chưa kiểm chứng vào thông báo lỗi hiển thị cho người dùng cuối.
- Tôi đề xuất xoá `STATUE_PALETTE[6]` (mục chết) — `mesh-holes` đo và chỉ ra xoá sẽ âm thầm đổi
  màu đế tượng từ xám sang đỏ vì tra cứu theo `pid % len` → Bài học: không xoá phần tử giữa một
  mảng được tra bằng modulo mà không kiểm tác động toàn downstream.
