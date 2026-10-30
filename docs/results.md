# Kết quả chạy Android — 30/10/2026

Đo lúc 16:43:58 (Asia/Bangkok) trên **Android 11 AOSP x86_64 giả lập**, LiteRT 1.4.2; `isEmulator=true`. Khôi phục, YOLO Detection, YOLO Segmentation và tính AP đều chạy trong app Android. **Không phải số đo Redmi.**

Bộ bổ sung: ba cảnh COCO val2017 có nhãn instance gốc, mỗi cảnh tạo năm suy thoái thời tiết bằng seed cố định → 15 ảnh, ba ảnh/loại. Đây không phải test split restoration gốc của GridFormer và không đủ để kết luận khả năng tổng quát hóa. Không dùng dự đoán làm nhãn.

| Loại | Detection mAP50:95 trước → sau | Δ điểm % | Segmentation mAP50:95 trước → sau | Δ điểm % |
| --- | ---: | ---: | ---: | ---: |
| Mưa | 30.13% → 31.22% | +1.09 | 30.40% → 32.07% | +1.67 |
| Sương | 39.39% → 45.09% | +5.70 | 35.20% → 41.86% | +6.66 |
| Mưa + sương | 24.54% → 27.90% | +3.37 | 13.53% → 24.82% | +11.30 |
| Tuyết | 34.77% → 37.81% | +3.03 | 35.37% → 35.65% | +0.28 |
| Đa thời tiết | 19.24% → 20.36% | +1.12 | 11.75% → 16.19% | +4.44 |
| Toàn bộ | 29.14% → 29.49% | +0.35 | 23.91% → 27.51% | +3.59 |

`ALL` được tính lại từ ranking confidence toàn tập theo từng class, không lấy trung bình các hàng. AP dùng 101 điểm recall, IoU 0,50:0,05:0,95, class macro; không có crowd/ignore/area ranges. JSON gốc có cả mAP50, số nhãn, số prediction và AP từng class.

## Bằng chứng

- [JSON mAP Android](../dist/validation/emulator-5580/evaluation.json)
- [Kiểm tra runtime và kích thước ảnh](../dist/validation/emulator-5580/device-validation.json)
- [Ảnh 31×19 và 257×259](../dist/validation/emulator-5580/edge-size-validation.json)
- [Đối chiếu pixel Android/host](../dist/validation/emulator-5580/host-android-comparison.json)
- [Execution plan native, CPU output không đổi](../dist/validation/emulator-5580/native-probe-validation.json)
- [YOLO và trường hợp GPU giả lập không khả dụng](../dist/validation/emulator-5580/yolo-device-validation.json)
- [Giao diện so sánh](../dist/validation/emulator-5580/11-comparison-slider.png)
- [Giao diện mAP](../dist/validation/emulator-5580/13-evaluation-results.png)
- [Ba model trong app](../dist/validation/emulator-5580/15-models-bundled.png)
- [Video thao tác app, không âm thanh](../dist/validation/emulator-5580/walkthrough.mp4)

GPU giả lập khởi tạo thất bại ở OpenCL/EGL; lỗi được lưu, không coi các node dự kiến hỗ trợ là bằng chứng GPU đã chạy. Native observer đã đọc đúng 256 node YOLO trước/sau và không đổi đầu ra CPU. GPU toàn phần/kết hợp trên Redmi và phép đo 5+50 trên điện thoại **được hoãn theo yêu cầu người dùng**, chưa có số liệu.

## Ảnh đối chiếu

Cột: ảnh suy thoái → ảnh khôi phục thực sự lấy từ app Android → ảnh clean gốc để đối chiếu bằng mắt. Các ảnh clean không dùng làm nhãn giả.

![COCO 285: degraded, Android restored, clean](../dist/validation/emulator-5580/comparison-285.png)

![COCO 724: degraded, Android restored, clean](../dist/validation/emulator-5580/comparison-724.png)

![COCO 776: degraded, Android restored, clean](../dist/validation/emulator-5580/comparison-776.png)
