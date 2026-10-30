# Kiểm chứng và bàn giao

Ngày triển khai: 30/10/2026. Đã hoàn thành chuyển ba model FP32 và kiểm chứng chức năng trên **Android 11 AOSP x86_64 giả lập**. **Chưa nghiệm thu hiệu năng/GPU trên Redmi**, được để sau theo yêu cầu người dùng. Bộ test có nhãn là bộ bổ sung COCO val2017 với thời tiết tổng hợp, không phải test split restoration gốc của bài báo.

| Hạng mục | Bằng chứng / trạng thái |
| --- | --- |
| Checkpoint / strict load / FP32 | Đạt, ba model, báo cáo hash và sai số trong conversion-results.md |
| JVM tests / lint | 18 test đạt; lint 0 lỗi (còn cảnh báo phiên bản/phụ thuộc) |
| Android YOLO và native observer | Hai test instrumentation đạt; trước/sau 256 node CPU, output không đổi |
| Android toàn luồng | Một test đạt, 15 ảnh, Detection và Segmentation độc lập, JSON mAP thật |
| Ảnh nhỏ / cạnh lẻ | 31×19 và 257×259 qua inference thật, ghép đúng kích thước |
| Pixel Android so với host | 99,99944% kênh giống hệt; tối đa một mức màu 8-bit khác biệt |
| Giao diện | Đã xem screenshot bốn tab và video thao tác giả lập, không âm thanh (app không có audio) |
| GPU giả lập | Không khởi tạo được OpenCL/EGL; ghi lỗi, không xem là GPU đã chạy |
| Redmi 5 warm-up +50 đo / GPU plan | Chưa chạy theo yêu cầu hoãn của người dùng |

Kết quả số và các ảnh đối chiếu ở [results.md](results.md). Bằng chứng gốc ở `dist/validation/emulator-5580`. Không đưa số liệu giả lập thành số đo điện thoại.

## Kiểm tra tự động

Chạy tại gốc dự án:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

Giữ test XML/HTML, lint HTML và build log trong `dist/validation`. JVM test kiểm tra coverage ô, ghép identity, NMS/letterbox, mAP xếp hạng confidence toàn tập, matching một-một, sweep IoU, class macro, ảnh âm tính, không có nhãn và mask IoU. Không chứng minh ảnh thật phục hồi tốt hoặc GPU hoạt động.

## Các bước phải hoàn tất trên Android

1. Chuyển ba mô hình bằng tools, giữ `.validation.json`, môi trường pip và log; so sánh PyTorch/TFLite trên ảnh thật. Cài APK trên thiết bị, nhập mô hình và kiểm tra không có unsupported ops/non-finite output. Giữ model hash.
2. Khôi phục ảnh <256, đúng256, kích thước lẻ và lớn hơn256; xác nhận kích thước gốc, biên ảnh và đường nối. Chụp màn hình cả trước/sau, kiểm tra xoay ảnh EXIF và xuất PNG mở lại được.
3. Nhập ZIP thật có nhãn cho năm loại đã thống nhất, vài ảnh mỗi loại. Chạy hai tác vụ độc lập; giữ JSON toàn bộ/từng loại, ảnh khôi phục và nhãn. mAP có thể giảm; báo cáo đúng giá trị delta, không mặc định phải tăng.
4. Benchmark khi thiết bị ổn định: lưu model hash, hãng/model, ABI, phiên bản Android, cấu hình luồng (4), pin/thermal, ít tác vụ nền. Mỗi backend 5 warm-up +50 lần đo, giữ đủ50 mẫu và trung bình trên app. Đo tile inference khác với tổng thời gian phục hồi ảnh.
5. App đã có native observing delegate trước/sau GPU, đọc `TfLiteContext.GetExecutionPlan` và `TfLiteDelegateParams.nodes_to_replace`. Không thay graph/output. Chỉ phân loại khi số node gốc khớp GPU nodes +CPU nodes còn lại: GPU_FULL (CPU=0), GPU_MIXED (có cả hai), CPU_ONLY (không có GPU partition), hoặc UNVERIFIED nếu chưa khớp. Lỗi khởi tạo là UNAVAILABLE. Tắt XNNPACK cho GPU request để không nhầm delegate CPU với GPU. Native ABI/quan sát CPU đã qua test; phân bố GPU trên Redmi cần thực thi thật. Một model/thiết bị không nhất thiết có đủ cả ba loại GPU để tạo ba bảng thời gian.
6. Test GPU không hỗ trợ, sai model/task/dtype/shape, ZIP đường dẫn lỗi, nhãn thiếu, ảnh quá lớn và bộ test thay thế; giữ thông báo lỗi. Quay video thao tác ứng dụng trên thiết bị, ghi logcat và ghi nhận crash/ANR nếu có.

Redmi Note 9 Pro Android 12 đã được phát hiện, nhưng người dùng chọn tiếp tục phần khác và đo điện thoại sau. Chưa có `benchmark.json` được tạo từ Redmi hoặc giả lập, không có số thời gian 5+50 được công bố trong bản giao này. Benchmark đã triển khai trong app; có test riêng chạy theo opt-in và không thay bộ test/nhãn.

Khi máy rảnh, có thể chạy benchmark bằng nút trong app hoặc test riêng (cần cài APK test trước):

```powershell
adb -s <serial> install -r app\build\outputs\apk\debug\app-debug.apk
adb -s <serial> install -r app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
adb -s <serial> shell am instrument -w -r -e class vn.edu.gridrestore.BenchmarkDeviceTest -e runBenchmark true vn.edu.gridrestore.test/androidx.test.runner.AndroidJUnitRunner
python tools\fetch_android_results.py --serial <serial> --output dist\validation\<serial>
```

Không chạy lệnh opt-in khi điện thoại đang dùng cho việc khác; cần giữ máy rảnh để số đo có ý nghĩa. Báo cáo có 50 mẫu, trung bình/std/min/max, hash input/model, thông tin pin/thermal/RAM và execution plan. Thu thêm logcat khi cần chẩn đoán driver.

## Gói nộp Teams

- ZIP mã nguồn có Gradle Wrapper, hướng dẫn build, scripts chuyển đổi, nguồn tải/hash mô hình và kết quả kiểm chứng.
- APK 0.2.0 kèm ba mô hình, model ZIP và test ZIP đã có SHA-256. Không cần chuyển đổi hoặc nhập model thủ công để demo.
- Ảnh test/nhãn theo từng loại, vài ảnh trước/sau, JSON mAP và benchmark, log GPU placement, screenshot/video.
- Text hướng dẫn sử dụng, thông tin nhóm, ID nhóm và ngày nộp. Tạo thư mục `BTL/<ID nhóm>` trên Teams; source/APK lớn nên dùng link tải theo yêu cầu môn học. Không tự upload/publish từ script.

Deadline là trước ngày thi cuối kỳ 7 ngày; cần điền ngày thi và ID nhóm. `tools/package.ps1` chỉ đóng gói cục bộ, xác minh hash model trong APK. Phần còn chờ theo lựa chọn người dùng là đo trên Redmi; khi nộp phải ghi đúng nguồn bộ test bổ sung và yêu cầu tập test của giảng viên.
