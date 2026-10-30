# GridRestore — BTL Lập trình Mobile

Ứng dụng Android Kotlin chạy **GridFormer [68]** và YOLO trực tiếp trên thiết bị. Nhóm: **Nguyễn Sỹ Vinh – Nguyễn Phạm Anh Tuấn – Lương Văn Hưng**.

## Trạng thái

Đã nạp checkpoint GridFormer chính thức với `strict=True`, chuyển **ba mô hình FP32 TFLite thật** và kiểm tra số học với PyTorch. APK 0.2.0 **đã kèm GridFormer, YOLO Detection, YOLO Segmentation** và bộ test bổ sung 15 ảnh có nhãn COCO gốc, năm nhóm thời tiết do người dùng xác nhận. Không cần tự nhập mô hình để bắt đầu. Kiểm tra Android giả lập và trạng thái nghiệm thu được ghi trong [docs/validation.md](docs/validation.md). **Đo trên Redmi được hoãn theo yêu cầu người dùng; chưa có số đo CPU/GPU điện thoại.**

- [Hướng dẫn mô hình và bằng chứng tải](docs/models.md)
- [Định dạng bộ test](docs/dataset.md)
- [Kiểm thử thiết bị và tiêu chí hoàn thành](docs/validation.md)

## Build và cài đặt

Yêu cầu JDK 17, Android SDK platform 36, build tools tương thích AGP, NDK `27.2.12479018`, CMake `3.22.1`. Android Studio có thể cài các SDK còn thiếu. Gradle Wrapper 8.14.3 / AGP 8.13.0 / Kotlin 2.2.10; `minSdk=26`, `targetSdk=36`. Chọn LiteRT **1.4.2 Interpreter API**, gồm runtime/API/GPU/GPU API; nhánh 2.x có thay đổi hỗ trợ GPU Interpreter nên không nâng đồng loạt theo số phiên bản.

Mở thư mục gốc bằng Android Studio, sync Gradle. Trên Windows:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Trên Linux/macOS dùng `./gradlew`. Đặt SDK bằng Android Studio hoặc `ANDROID_HOME`; `local.properties` chỉ dành cho máy cục bộ, không được đóng gói. Không cần API key hay server. APK debug đã ký bằng khóa debug cục bộ, phù hợp cài thử; không phải bản phát hành Play Store.

## Luồng sử dụng

1. **Mở app:** ba mô hình đi kèm được chép vào vùng riêng của app và kiểm SHA-256 ở lần đầu. **Thiết lập** vẫn cho nhập mô hình thay thế theo đúng contract; app kiểm tensor và chạy thử trước khi lưu. Xem [docs/models.md](docs/models.md).
2. **Khôi phục:** chọn ảnh tối đa 4 megapixel, chọn CPU hoặc GPU delegate, khôi phục rồi kéo thanh so sánh trước/sau và xuất PNG. Ảnh có EXIF được xoay đúng khi nhập. Ô 256×256 có overlap 32 px; mép ảnh lặp pixel, ghép theo trọng số về đúng kích thước gốc.
3. **Đánh giá:** nhập ZIP ảnh test có nhãn, chạy GridFormer rồi YOLO riêng cho Detection và Segmentation trên cả hai ảnh. Tính mAP50 và mAP50:95, delta theo loại suy thoái và toàn bộ tập; xuất JSON. Ground truth phải do người gán/kiểm chứng, cùng hệ tọa độ và class mapping YOLO.
4. **Hiệu năng:** chọn ảnh đầu vào, chạy cùng ô 256×256 trên CPU và thử GPU, mỗi cấu hình 5 warm-up + 50 lần đo. App hiển thị trung bình/min/max/độ lệch chuẩn và xuất 50 mẫu thô, SHA-256, thông tin thiết bị, nhiệt độ pin/thermal/RAM trước và sau. Bộ quan sát native đọc execution plan trước/sau GPU để phân loại toàn phần/kết hợp/CPU; chỉ kết luận khi số node khớp. Lỗi GPU được lưu riêng.

App không yêu cầu quyền truy cập toàn bộ bộ nhớ hoặc Internet. Android picker cấp quyền cho từng tệp. Dữ liệu nằm trong vùng riêng của app; gỡ app sẽ mất dữ liệu chưa xuất. Các ảnh test đã khôi phục nằm trong `files/evaluation-images/`; bản debug cho phép lấy bằng `adb exec-out run-as vn.edu.gridrestore ...`.

## Cấu trúc

```text
app/src/main/java/vn/edu/gridrestore/
  MainActivity.kt, ComparisonView.kt  Giao diện và luồng tác vụ
  inference/                         LiteRT, chia/ghép ô, YOLO, benchmark
  core/Evaluation.kt                 IoU, mask IoU, AP và accumulator
  data/                              ZIP/JSON, ảnh, thí nghiệm và báo cáo
app/src/test/                        Kiểm thử thuật toán JVM
tools/                               Tải checkpoint, chuyển đổi, contracts
docs/                                Hướng dẫn và trạng thái kiểm chứng
```

Chỉ giữ scalar confidence/TP giữa các ảnh khi tính AP, tránh giữ toàn bộ mask của dataset. YOLO sử dụng confidence 0.001, NMS theo lớp IoU 0.7, maxDet 100. AP dùng 101 điểm recall và IoU 0.50:0.05:0.95; không triển khai crowd/ignore/area-range của COCO đầy đủ. Khi hết ground truth, kết quả là N/A. Không lấy confidence thay mAP.

## Giới hạn cần giải quyết trước nộp

- GridFormer đã chuyển FP32 thành công. Trên hai ảnh thật, sai số tối đa so với PyTorch là `1.20e-6`. Ảnh đen có sai số tối đa `3.59e-4`, không đạt allclose ban đầu `atol=2e-4`; đã ghi rõ và kiểm thêm ngân sách ảnh 8-bit (max raw ≤0.5/255, PSNR≥70dB, max pixel sau clamp/round≤1). Không giấu kết quả allclose thất bại. Xem báo cáo model.
- Người dùng đã chọn: mưa, sương, mưa+sương, tuyết, đa thời tiết. Bộ test đi kèm lấy ba cảnh COCO **val2017** có nhãn gốc, áp dụng năm suy thoái tổng hợp có seed; đây là bộ bổ sung, không thay tập test restoration của bài báo. Cần giữ sự phân biệt này trong bài nộp.
- Bộ test đánh giá giới hạn 1 megapixel/ảnh và có kiểm tra ngân sách RAM cho mask. GridFormer lớn; thiết bị ít RAM có thể không chạy được dù đã khai báo largeHeap. Không tự giảm ảnh test vì sẽ làm sai nhãn.
- Bằng chứng Android giả lập tách riêng khỏi Redmi. Chưa có phép đo CPU/GPU trên điện thoại vì người dùng yêu cầu để sau; không suy ra hiệu năng điện thoại từ giả lập.
- `tools/package.ps1` tạo APK, source ZIP, model ZIP, test ZIP và checksum; kiểm hash model ngay trong APK trước khi đóng gói. Source ZIP không chứa trọng số lớn. Khi giải nén source: giải nén model ZIP vào `tools/exports`, test ZIP vào `tools/test-data` và chép test ZIP thành `tools/test-data.zip`, rồi chạy `python tools/bundle_assets.py` trước build. Deadline: trước thi cuối kỳ 7 ngày; ngày thi và ID nhóm chưa được cung cấp.

Nguồn: [GridFormer chính thức](https://github.com/TaoWangzj/GridFormer), [LiteRT Android](https://developers.google.com/edge/litert/android), [GPU delegate](https://developers.google.com/edge/litert/android/gpu). Giấy phép nguồn/mô hình bên thứ ba cần được giữ khi phân phối; xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
"# 2026.1_mobile" 
