# Mô hình và quy trình chuyển đổi

## Trạng thái kiểm chứng ngày 30/10/2026

Đã tải **checkpoint GridFormer Multi-Weather thật** từ liên kết Google Drive do tác giả công bố. File cục bộ nằm ở `tools/downloads/gridformer/net_g_best.pth` (thư mục này được bỏ qua khi commit).

| Kiểm tra | Kết quả |
| --- | --- |
| GitHub chính thức và cấu hình Multi-Weather | Truy cập được |
| Commit mã nguồn | `d0ec11c2bf3bb0f09e00059006523c4c44c535a0` |
| File ID trên Google Drive | `1aNyEQPHFviCYr_H_rrz8m3mqlRUGqEuR` |
| Tải toàn bộ checkpoint | Thành công, **243.056.049 byte** |
| SHA-256 | `5cdfaff392732e42856c42029889386cfe607af1c95f85d7f73ea00dee301fcc` |
| Kiểm tra CRC ZIP | Đạt, 2.990 entries |
| Đọc tên state dictionary bằng `pickletools` | Có `params` và `params_ema`; chưa thực thi pickle |
| Nạp PyTorch với `strict=True` | **Đạt**, key `params` |
| Xuất TFLite và đối chiếu số học | **Đạt ba mô hình**, xem [báo cáo chuyển đổi](conversion-results.md) |
| Android runtime / mAP | Bằng chứng giả lập trong `dist/validation/emulator-5580`, xem validation.md |
| Redmi / GPU điện thoại | **Để sau theo yêu cầu người dùng**, chưa có số đo |

Bằng chứng tải ban đầu nằm trong [`tools/download_verification.json`](../tools/download_verification.json); các cờ của file đó phản ánh thời điểm tải. Sau đó đã tạo môi trường WSL riêng và chuyển thành công ba mô hình. **APK 0.2.0 kèm các model FP32 thật**. Hash, sai số, giới hạn và quy trình đã chạy được ghi trong [conversion-results.md](conversion-results.md). Không có bộ lọc giả đóng vai GridFormer.

## 1. Tải lại nguồn và checkpoint

Từ thư mục gốc của dự án, dùng một môi trường Python riêng:

```powershell
python -m venv .venv-download
.venv-download\Scripts\python -m pip install -r tools/requirements-download.txt
.venv-download\Scripts\python tools/download_gridformer.py --output-dir models/source
```

Script tải mã kiến trúc và YAML từ commit cố định, tải checkpoint bằng ID đã kiểm tra, xác nhận dung lượng/CRC rồi ghi SHA-256 vào `download_manifest.json`. Script không ghi đè file đã có. Nếu Google Drive giới hạn lượt tải, tải thủ công từ [thư mục Multi-Weather chính thức](https://drive.google.com/drive/folders/1jLHWQwzUMb6rOmZ4S0vnIBnxfmh7sf4Z) và kiểm tra:

```powershell
python tools/download_gridformer.py --inspect-only C:\path\to\net_g_best.pth
```

`--inspect-only` chỉ kiểm tra checkpoint; nó không tự tạo bộ nguồn và manifest cần cho bước export. Có thể dùng bộ `tools/downloads/gridformer` đã tải trong workspace này làm `--source-dir`.

## 2. Xuất FP32 trên Linux

Dùng **Linux x86_64, Python 3.11**, chẳng hạn WSL2 hoặc Linux/Colab. Chạy từ bản sao đầy đủ của dự án. Giữ riêng môi trường chuyển đổi với môi trường Android; GPU không phải điều kiện bắt buộc của script. Nên dùng máy có đủ RAM cho checkpoint và graph GridFormer lớn.

```bash
bash tools/setup_linux.sh
source "$HOME/.local/share/gridrestore/venv/bin/activate"

# Nếu chưa tải bằng bước 1:
"$HOME/.local/share/gridrestore/bin/uv" pip install --python "$HOME/.local/share/gridrestore/venv/bin/python" -r tools/requirements-download.txt
python tools/download_gridformer.py --output-dir models/source

python tools/export_models.py --kind gridformer \
  --source-dir models/source --output-dir tools/exports \
  --sample-image /path/to/real_test_image.png --restoration-display-budget

python tools/export_models.py --kind yolo --output-dir tools/exports
```

Để dùng checkpoint đã được tải trong workspace, thay `--source-dir models/source` bằng `--source-dir tools/downloads/gridformer` và không chạy lại lệnh download. Dùng `--sample-image` nhiều lần để kiểm tra thêm ảnh thật.

Toolchain đầy đủ đã khóa trong `tools/requirements-convert-lock.txt`. Dùng `bash tools/setup_linux.sh` để cài môi trường CPU riêng, không thay Python/CUDA hệ thống. Quy trình thực tế gồm sửa cách biểu diễn toán tử, inline graph, tiếp tục từ SavedModel và kiểm tra candidate với PyTorch gốc; log được giữ trong `tools/work`. Với GridFormer thêm `--restoration-display-budget` như mô tả trong báo cáo chuyển đổi; YOLO vẫn dùng allclose mặc định.

### Các chốt kiểm tra của exporter

1. Kiểm tra hash nguồn/checkpoint, nạp `state_dict` với `strict=True`.
2. Bọc NCHW của PyTorch thành NHWC cho Android, lấy output khôi phục ở độ phân giải đầy đủ.
3. Xuất `.candidate.tflite`, mở lại bằng **Interpreter độc lập**, kiểm tra dtype và toàn bộ shape.
4. So sánh PyTorch/TFLite trên ảnh đen, xám, nhiễu với seed cố định và các ảnh thật được cung cấp. Mặc định `atol=0.0002`, `rtol=0.002`; ghi sai số thực tế vào JSON. Kiểm tra tensor FP16 để tránh gắn nhãn FP32 sai.
5. Chỉ khi tất cả kiểm tra đạt mới đổi tên sang `.tflite` cuối cùng và ghi `.validation.json` gồm hash, package versions, operators và kết quả số học.

Ứng dụng dùng TensorFlow Lite Interpreter trên Android; một export mở được trên runtime Python mới vẫn cần được kiểm tra trên runtime Android của dự án. Nếu Android báo toán tử hoặc phiên bản không hỗ trợ, đó là lỗi tích hợp chưa giải quyết. Không thay mô hình khác hay Select TF Ops một cách im lặng.

### Chi tiết GridFormer cần lưu ý

- YAML chính thức khai báo `type: GridFormer_v15`, trong khi file kiến trúc tại commit đã ghim chỉ định nghĩa lớp **`GridFormer`**. Script dùng trực tiếp lớp này, chỉ bỏ import/decorator đăng ký BasicSR, giữ nguyên phép toán, và bắt buộc strict load để phát hiện sai kiến trúc.
- Cấu hình mạng sử dụng `dim=48`, `num_blocks=[1,2,4]`, `height=3`, `width=6`, `heads=[2,4,8]`, `sample_rate_trunk=[4,2,2]`.
- Mặc định lấy key **`params`**, đúng mặc định loader kiểm thử của BasicSR. `--state-key params_ema` là thí nghiệm riêng và được ghi nhận trong manifest.
- `GridFormer.forward()` đảo thứ tự ba ảnh output; exporter lấy phần tử cuối để nhận ảnh 256×256. Ảnh RGB đầu vào chuẩn hóa `[0,1]`; app kẹp kết quả về `[0,1]` khi dựng bitmap.
- Kiểm tra tương đương số học không chứng minh chất lượng phục hồi. Cần chạy tập test gốc và các nhãn YOLO tương ứng trên Android.

## 3. Hợp đồng model với app Android

Chọn chức năng nhập mô hình trong app cho từng file tương ứng. Mô hình nằm trong bộ nhớ riêng của app, không được gửi lên máy chủ để suy luận.

| File nhập | Input FLOAT32 | Output FLOAT32 |
| --- | --- | --- |
| `gridformer.tflite` | `[1,256,256,3]` RGB NHWC `[0,1]` | `[1,256,256,3]` NHWC |
| `yolo_detect.tflite` | `[1,640,640,3]` RGB NHWC `[0,1]` | `[1,84,8400]` |
| `yolo_segment.tflite` | `[1,640,640,3]` RGB NHWC `[0,1]` | `[1,116,8400]` và `[1,160,160,32]` |

Hợp đồng máy đọc được nằm trong [`tools/model_contract.json`](../tools/model_contract.json). File này mô tả **yêu cầu**, không phải bằng chứng đã export.

YOLO dùng **YOLOv8n / YOLOv8n-seg COCO 80 lớp**; không nhầm với YOLOv5 có objectness hoặc các bản end-to-end đã gộp NMS. Exporter đặt head `format="tflite"` để nhận **xywh chuẩn hóa theo khung letterbox 640×640**, theo mã head của Ultralytics 8.3.160. Detection có 4 tọa độ + 80 điểm lớp; segmentation thêm 32 hệ số mask. Prototype được chuyển sang NHWC. NMS theo lớp, giải mã mask và hoàn tác letterbox do app thực hiện. Thứ tự lớp ground truth phải khớp 80 lớp COCO theo metadata của model, không dùng trực tiếp category ID COCO không liên tiếp.

Xuất YOLO bằng wrapper LiteRT Torch trong script này nhằm giữ chính xác ABI kể trên. Không trộn model xuất từ lệnh Ultralytics khác khi chưa kiểm tra shape/normalization. Ultralytics hiện đã chuyển hướng tài liệu `format="tflite"` sang `litert`; phiên bản và wrapper trong dự án được ghim để tránh API mới làm đổi ABI.

## 4. Phạm vi năm loại thử nghiệm

Theo README GridFormer, năm **tác vụ** là khử mưa, khử sương, khử mưa kèm sương, khử tuyết và phục hồi đa thời tiết. Chúng không đồng nghĩa với bộ năm loại denoising/deblurring/low-light/dehazing/deraining. Trọng số Multi-Weather được công bố cho dữ liệu hỗn hợp liên quan tuyết, giọt mưa và mưa/sương. Nhóm cần ghi rõ cách ánh xạ các nhóm test của bài báo sang yêu cầu năm loại của môn học; không báo cáo đã kiểm thử các loại suy thoái ngoài phạm vi checkpoint khi chưa có dữ liệu và kết quả.

Để tính mAP, cần nhãn ground truth detection/instance segmentation cho **cùng cảnh, cùng kích thước tọa độ** trước và sau phục hồi. Dataset phục hồi có ảnh clean/degraded không tự động có nhãn đối tượng. Không dùng dự đoán YOLO từ ảnh clean làm ground truth rồi coi là mAP thực. Không lấy confidence score thay mAP.

Benchmark phải thực hiện trên thiết bị thật: warm-up 5 lần, đo 50 lần và lưu từng mẫu. Trạng thái khởi tạo GPU delegate thành công chưa đủ chứng minh toàn bộ graph chạy GPU. Phải giữ log/profiler về các node được delegate trước khi phân loại GPU hoàn toàn hay GPU + CPU; kết quả chưa đo cần ghi đúng là chưa xác minh.

## Nguồn chính thức

- [TaoWangzj/GridFormer](https://github.com/TaoWangzj/GridFormer): nguồn mô hình, dữ liệu, pretrained weights.
- [Cấu hình kiểm thử Multi-Weather tại commit đã ghim](https://github.com/TaoWangzj/GridFormer/blob/d0ec11c2bf3bb0f09e00059006523c4c44c535a0/options/test/GridFormer/Multi_weather_Restoration/GridFormer_mixed.yml).
- [Kiến trúc GridFormer chính thức tại commit đã ghim](https://github.com/TaoWangzj/GridFormer/blob/d0ec11c2bf3bb0f09e00059006523c4c44c535a0/basicsr/archs/GridFormer_arch.py).
- [Google LiteRT Torch v0.8.0: yêu cầu Linux và API chuyển đổi](https://github.com/google-ai-edge/litert-torch/tree/v0.8.0).
- [Google: chuyển PyTorch sang LiteRT và đối chiếu kết quả](https://developers.google.com/edge/litert/conversion/pytorch/overview).
- [Ultralytics YOLO head v8.3.160: tọa độ TFLite chuẩn hóa](https://github.com/ultralytics/ultralytics/blob/v8.3.160/ultralytics/nn/modules/head.py).
- [Ultralytics: thay đổi TFLite/LiteRT](https://docs.ultralytics.com/integrations/tflite/). Giữ attribution và tuân thủ giấy phép AGPL-3.0 của mô hình/phần mềm Ultralytics khi phân phối.
