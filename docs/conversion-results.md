# Kết quả chuyển đổi mô hình — 30/10/2026

Đã nạp checkpoint GridFormer chính thức bằng `torch.load(weights_only=True)` và `load_state_dict(strict=True)`, key `params`. Không huấn luyện lại hoặc đổi trọng số. APK 0.2.0 kèm ba model dưới đây, tự kiểm SHA-256 khi chép vào vùng riêng của app.

| Model | Byte | SHA-256 |
| --- | ---: | --- |
| GridFormer | 120684168 | `4b073ef90349d2d97d04a7339775c3be5fda520da8be7bda703f423943f61d6c` |
| YOLOv8n detection | 12836808 | `59b4d0fb78b0b04644d0dab144fc72da18edfd73e95e90f6e6537c7f1d72d0fe` |
| YOLOv8n-seg | 13870960 | `694e716fd2a2c007e673cf70eafcc8aa8c13b7e2b46441ef973e2ffb41a36c0d` |

Mỗi model có `.validation.json` trong `tools/exports`, assets APK và model ZIP. Input/output là FLOAT32, không có tensor FP16. Restoration dùng NHWC `[1,256,256,3]`, RGB `[0,1]`. YOLO dùng input NHWC 640×640, raw normalized xywh, NMS ngoài model. Thứ tự output segmentation có thể là prototype trước; app nhận diện theo shape, không mặc định index 0 là detection.

## Đối chiếu số học

Mỗi model được đối chiếu trên ảnh đen, xám, nhiễu có seed, COCO val2017 ảnh 285 và ảnh haze tổng hợp từ ảnh 724. YOLO đạt allclose `atol=0.0002, rtol=0.002` trên cả năm mẫu.

| Model | Sai số tuyệt đối lớn nhất |
| --- | ---: |
| YOLO detection | 8.8215e-6 |
| YOLO segmentation | 2.3127e-5 |
| GridFormer, hai ảnh thật | 1.1921e-6 |
| GridFormer, tất cả mẫu | 3.5889e-4 (ảnh đen) |

**GridFormer không đạt allclose ban đầu trên ảnh đen:** 87/196608 phần tử (0,0443%) vượt ngưỡng, max error 0,00035889. Kết quả `strict_allclose_passed=false` và log thất bại được giữ. Các mẫu còn lại đạt allclose, bao gồm hai ảnh thật.

Vì app hiển thị RGB 8-bit sau clamp/round, lần kiểm tra cuối dùng ngân sách ảnh được ghi rõ bằng `--restoration-display-budget`: max raw error ≤0,5/255; MSE ≤1e-7 (PSNR giữa hai runtime ≥70 dB); sai khác kênh 8-bit ≤1. Cả năm mẫu đạt; PSNR thấp nhất **87,10 dB**, max sai khác lượng tử hóa **một mức màu**. Đây là đánh giá tương đương runtime, **không phải PSNR phục hồi so với ảnh clean** và không phải tuyên bố bitwise giống nhau.

## Các lỗi chuyển đổi đã xử lý

1. YAML gọi `GridFormer_v15` trong khi nguồn định nghĩa `GridFormer`. Dùng trực tiếp lớp chính thức và bắt buộc strict load.
2. LiteRT Torch 0.8.0 không hạ đúng depthwise ConvTranspose2d. Với kernel=stride, padding/output_padding=0, dilation=1, phép toán tương đương depthwise conv 1×1 mở rộng kênh + pixel shuffle. Đổi cách biểu diễn **117 module** sau strict load, kiểm tra từng module trên tensor có seed. So sánh toàn mạng dùng **các module gốc chưa đổi**, không so bản rewrite với chính nó.
3. Tên hàm JAX bridge ngẫu nhiên bị cắt còn 32-bit, gây va chạm trên graph lớn. Dùng namespace tăng đơn điệu, chỉ đổi tên symbol.
4. Có 22834 helper/model làm TensorFlow nhập StableHLO rất chậm. Inline bằng inliner của converter, verify IR; cache SavedModel thực tế chứa hai module và cả hai đã được inline. TFLite cuối có 14401 toán tử trước runtime delegation.
5. Thứ tự output segmentation bị đảo so với tuple PyTorch. Kiểm tra/so sánh theo shape và giữ actual output order trong báo cáo.

## Tái lập

Linux x86_64, CPython 3.11.14, Torch 2.8.0+cpu, LiteRT Torch 0.8.0, AI Edge TensorFlow 2.21.0.dev20251110, LiteRT Python 2.1.0. Toàn bộ phiên bản trong `tools/requirements-convert-lock.txt`.

```bash
bash tools/setup_linux.sh
PY="$HOME/.local/share/gridrestore/venv/bin/python"
UV="$HOME/.local/share/gridrestore/bin/uv"
"$UV" pip install --python "$PY" -r tools/requirements-download.txt
"$PY" tools/download_gridformer.py --output-dir models/source
"$PY" tools/prepare_test_data.py --count 3 --max-side 256
OMP_NUM_THREADS=2 "$PY" tools/export_models.py --kind gridformer \
  --source-dir models/source --output-dir tools/exports \
  --sample-image tools/test-data/clean/285.png \
  --sample-image tools/test-data/images/haze_724.png --restoration-display-budget
OMP_NUM_THREADS=2 "$PY" tools/export_models.py --kind yolo \
  --output-dir tools/exports --sample-image tools/test-data/clean/285.png \
  --sample-image tools/test-data/images/haze_724.png
python tools/bundle_assets.py
```

Script xuất hiện tại đã tích hợp namespace và inline. Phiên thực tế đã tiếp tục từ cache bằng các lệnh sau; giữ lại để chẩn đoán/tái lập đường đi đã kiểm chứng:

```bash
"$PY" tools/inline_saved_model.py tools/work/gridformer-saved-model tools/work/gridformer-inlined-saved-model
OMP_NUM_THREADS=2 "$PY" tools/convert_saved_model.py tools/work/gridformer-inlined-saved-model tools/exports/gridformer.candidate.tflite
OMP_NUM_THREADS=2 "$PY" tools/export_models.py --kind gridformer \
  --source-dir tools/downloads/gridformer --output-dir tools/exports \
  --sample-image tools/test-data/clean/285.png \
  --sample-image tools/test-data/images/haze_724.png --reuse-candidate --restoration-display-budget
```

Candidate chỉ đổi sang tên cuối sau đối chiếu; không đổi tên thủ công để bỏ qua kiểm tra. Các cache lớn không nằm trong source ZIP. Model ZIP đủ để build lại APK mà không cần chuyển đổi lại. Kiểm chứng GPU và thời gian trên Redmi được để sau theo yêu cầu người dùng; không suy ra từ host hoặc giả lập.
