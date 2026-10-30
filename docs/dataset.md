# Bộ test và ground truth

APK 0.2.0 có nút **Dùng 15 ảnh test đi kèm**. Bộ này gồm ba cảnh COCO val2017 và năm dạng thời tiết tổng hợp theo lựa chọn người dùng, nhãn instance gốc từ COCO. Xem [kết quả chạy Android](results.md). Đây là dữ liệu bổ sung để kiểm chứng app; phân biệt với tập test restoration gốc của GridFormer.

App nhập một ZIP có **manifest.json ở thư mục gốc**. Đường dẫn dùng `/`, không có ổ đĩa hoặc `..` vượt ra ngoài thư mục. Giới hạn: 200 ảnh, 512 mục ZIP, 64 MB/tệp, 512 MB sau giải nén. File hỏng/ZIP trùng đường dẫn bị từ chối; không thay bộ test cũ khi manifest không đọc được.

```text
manifest.json
images/rain_001.png
images/haze_001.png
...
```

Ví dụ cấu trúc (tọa độ minh họa, không phải nhãn hay kết quả thí nghiệm thật):

```json
{
  "schemaVersion": 1,
  "name": "GridFormer test subset v1",
  "images": [
    {
      "id": "rain_001",
      "degradation": "rain",
      "file": "images/rain_001.png",
      "annotations": [
        {
          "classId": 0,
          "box": [20, 30, 100, 200],
          "polygons": [[[20,30],[100,30],[100,200],[20,200]]]
        }
      ]
    }
  ]
}
```

- `id` duy nhất, không rỗng. `degradation` là tên nhóm tự chọn, trừ `ALL` dành cho tổng hợp.
- `file` là ảnh suy thoái thực sự. Dùng PNG đã chuẩn hóa hướng; nhãn dựa trên kích thước pixel được mã hóa. Không dùng EXIF để xoay ảnh trong bộ test.
- `annotations` bắt buộc; dùng `[]` cho ảnh thực sự không có đối tượng. Không biến ảnh chưa gán nhãn thành ảnh âm tính.
- Mỗi nhãn là một **instance** với `classId` đúng index lớp mô hình. Với YOLOv8n COCO-80, `0=person`, không dùng COCO category_id gốc (có khoảng trống). Cùng class mapping phải dùng cho cả Detection và Segmentation.
- `box=[x1,y1,x2,y2]` theo pixel gốc, cạnh phải/dưới exclusive; diện tích dương và nằm trong ảnh. `polygons` là danh sách đa giác của cùng instance, gộp foreground; mỗi đa giác ít nhất ba điểm `[x,y]`. Không hỗ trợ lỗ/crowd/ignore/RLE ở bản này.
- Đánh giá chạy cả hai tác vụ nên mọi instance phải có polygon. App rasterize không anti-alias, tính IoU mask nhị phân theo pixel gốc. Ảnh sau khôi phục giữ kích thước nên dùng lại nhãn, không dịch/scaling nhãn khi chạy app.
- Ảnh đánh giá tối đa 1.048.576 pixel. Nếu số mask có thể vượt ngân sách RAM, app từ chối với thông báo giảm ảnh **và biến đổi nhãn tương ứng ở khâu chuẩn bị**. Nên bắt đầu 256–512 px, vài ảnh mỗi loại.

Chuẩn bị ít nhất năm nhóm được giảng viên chấp thuận, mỗi nhóm vài ảnh (báo cáo có cờ tối thiểu 2 ảnh/nhóm). Phải ghi lại nguồn dataset, split test, ID ảnh và cách gán nhãn. Cờ đủ năm nhóm chỉ kiểm đếm dữ liệu, không xác nhận phạm vi học thuật phù hợp. Đừng nhân bản cùng ảnh rồi đổi tên để đạt số lượng.

Ảnh restoration clean/degraded không tự có nhãn detection/segmentation. Có thể gán nhãn thủ công và kiểm tra trên cùng cảnh. Nếu tạo suy thoái trên dataset có nhãn, cần ghi rõ đây là bộ test bổ sung và xin giảng viên thống nhất khi thay thế tập test bài báo. Dự đoán từ ảnh clean không phải ground truth để công bố mAP thật.

Không dùng bộ test giả hoặc prediction làm ground truth. Bộ bổ sung đi kèm có suy thoái tổng hợp được công bố rõ nguồn/seed, nhãn COCO gốc và kết quả thực từ Android. Các unit test hình học chỉ kiểm tra thuật toán, không được đưa vào kết quả chất lượng BTL.
