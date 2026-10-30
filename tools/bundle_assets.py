"""Bundle only numerically validated models and a labelled supplemental dataset."""
from pathlib import Path
import hashlib, json, shutil

root=Path(__file__).resolve().parent.parent
exports=root/"tools/exports"
destination=root/"app/src/main/assets/models"
destination.mkdir(parents=True,exist_ok=True)
manifest={"schemaVersion":1,"models":{}}
for name in ("gridformer","yolo_detect","yolo_segment"):
    model=exports/(name+".tflite")
    validation=exports/(name+".validation.json")
    report=json.loads(validation.read_text())
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    assert report["host_cpu_validation_passed"] and report["sha256"]==sha, name
    shutil.copy2(model,destination/model.name)
    shutil.copy2(validation,destination/validation.name)
    manifest["models"][name]={"sha256":sha,"bytes":model.stat().st_size}
(destination/"manifest.json").write_text(json.dumps(manifest,indent=2))
shutil.copy2(root/"tools/test-data.zip",destination.parent/"sample-test.zip")
shutil.copy2(root/"tools/test-data/images/rain_285.png",destination.parent/"sample-input.png")
print(json.dumps(manifest,indent=2))
