"""Create a report and contact sheets from measured Android outputs only."""
from pathlib import Path
import json
from datetime import datetime,timezone,timedelta
from PIL import Image,ImageDraw,ImageFont,ImageOps

root=Path('.')
evidence=root/'dist/validation/emulator-5580'
report=json.loads((evidence/'evaluation.json').read_text())
manifest=json.loads((root/'tools/test-data/manifest.json').read_text())
for name,info in report['models'].items():
    host=json.loads((root/f'tools/exports/{name}.validation.json').read_text())
    assert info['sha256']==host['sha256'],name
labels={'rain':'Mưa','haze':'Sương','rain_haze':'Mưa + sương','snow':'Tuyết','multi_weather':'Đa thời tiết','ALL':'Toàn bộ'}
when=datetime.fromisoformat(report['createdAt'].replace('Z','+00:00')).astimezone(timezone(timedelta(hours=7)))
lines=['# Kết quả chạy Android — 30/10/2026','',f"Đo lúc {when:%H:%M:%S} (Asia/Bangkok) trên **Android 11 AOSP x86_64 giả lập**, LiteRT 1.4.2; `isEmulator=true`. Khôi phục, YOLO Detection, YOLO Segmentation và tính AP đều chạy trong app Android. **Không phải số đo Redmi.**",'', 'Bộ bổ sung: ba cảnh COCO val2017 có nhãn instance gốc, mỗi cảnh tạo năm suy thoái thời tiết bằng seed cố định → 15 ảnh, ba ảnh/loại. Đây không phải test split restoration gốc của GridFormer và không đủ để kết luận khả năng tổng quát hóa. Không dùng dự đoán làm nhãn.', '', '| Loại | Detection mAP50:95 trước → sau | Δ điểm % | Segmentation mAP50:95 trước → sau | Δ điểm % |','| --- | ---: | ---: | ---: | ---: |']
for row in report['results']:
    d=row['detection'];s=row['segmentation']
    a=d['before']['mAP50_95']*100;b=d['after']['mAP50_95']*100;c=s['before']['mAP50_95']*100;e=s['after']['mAP50_95']*100
    lines.append(f"| {labels[row['degradation']]} | {a:.2f}% → {b:.2f}% | {b-a:+.2f} | {c:.2f}% → {e:.2f}% | {e-c:+.2f} |")
lines+=['','`ALL` được tính lại từ ranking confidence toàn tập theo từng class, không lấy trung bình các hàng. AP dùng 101 điểm recall, IoU 0,50:0,05:0,95, class macro; không có crowd/ignore/area ranges. JSON gốc có cả mAP50, số nhãn, số prediction và AP từng class.','', '## Bằng chứng','', '- [JSON mAP Android](../dist/validation/emulator-5580/evaluation.json)', '- [Kiểm tra runtime và kích thước ảnh](../dist/validation/emulator-5580/device-validation.json)', '- [Ảnh 31×19 và 257×259](../dist/validation/emulator-5580/edge-size-validation.json)', '- [Đối chiếu pixel Android/host](../dist/validation/emulator-5580/host-android-comparison.json)', '- [Execution plan native, CPU output không đổi](../dist/validation/emulator-5580/native-probe-validation.json)', '- [YOLO và trường hợp GPU giả lập không khả dụng](../dist/validation/emulator-5580/yolo-device-validation.json)', '- [Giao diện so sánh](../dist/validation/emulator-5580/11-comparison-slider.png)', '- [Giao diện mAP](../dist/validation/emulator-5580/13-evaluation-results.png)', '- [Ba model trong app](../dist/validation/emulator-5580/15-models-bundled.png)', '- [Video thao tác app, không âm thanh](../dist/validation/emulator-5580/walkthrough.mp4)', '', 'GPU giả lập khởi tạo thất bại ở OpenCL/EGL; lỗi được lưu, không coi các node dự kiến hỗ trợ là bằng chứng GPU đã chạy. Native observer đã đọc đúng 256 node YOLO trước/sau và không đổi đầu ra CPU. GPU toàn phần/kết hợp trên Redmi và phép đo 5+50 trên điện thoại **được hoãn theo yêu cầu người dùng**, chưa có số liệu.', '', '## Ảnh đối chiếu', '', 'Cột: ảnh suy thoái → ảnh khôi phục thực sự lấy từ app Android → ảnh clean gốc để đối chiếu bằng mắt. Các ảnh clean không dùng làm nhãn giả.']
try:font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
except OSError:font=ImageFont.load_default()
for image in manifest['provenance']['images']:
    image_id=image['cocoImageId'];sheet=Image.new('RGB',(816,1560),(16,24,32));draw=ImageDraw.Draw(sheet)
    draw.text((16,12),f'COCO val2017 / {image_id} - supplemental synthetic weather',font=font,fill='white')
    for x,title in [(16,'DEGRADED'),(284,'ANDROID RESTORED'),(552,'CLEAN REFERENCE')]:draw.text((x,44),title,font=font,fill=(128,229,188))
    entries=[(index,sample) for index,sample in enumerate(manifest['images']) if sample['id'].endswith('_'+str(image_id))]
    for row,(index,sample) in enumerate(entries):
        y=84+row*292;draw.text((16,y),sample['degradation'],font=font,fill='white')
        paths=[root/'tools/test-data'/sample['file'],evidence/'evaluation-images'/f'sample_{index+1}.png',root/'tools/test-data'/sample['cleanFile']]
        for column,path in enumerate(paths):
            with Image.open(path) as bitmap:
                thumb=ImageOps.contain(bitmap.convert('RGB'),(256,256))
                sheet.paste(thumb,(16+column*268+(256-thumb.width)//2,y+26))
    output=evidence/f'comparison-{image_id}.png';sheet.save(output)
    lines+=['',f"![COCO {image_id}: degraded, Android restored, clean](../dist/validation/emulator-5580/{output.name})"]
(root/'docs/results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote docs/results.md and three contact sheets from Android evidence')
